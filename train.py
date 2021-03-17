"""
Training/evaluation class
"""
import tensorflow.compat.v2 as tf
from model import RecoveryLikelihood
import datasets
from eval_utils import *
from train_utils import *
import time
import pickle
import pygrid


class Trainer:
  def __init__(self, *, hps):
    super(Trainer, self).__init__()
    self.hps = hps

  @tf.function
  def init_opt(self):
    x = tf.random.normal([2, self.hps.img_sz, self.hps.img_sz, 3])
    with tf.GradientTape() as tape:
      tape.watch(self.diffusion.trainable_variables)
      loss = tf.reduce_sum(self.diffusion.net(x, 0, dropout=0.))
    vars = self.diffusion.trainable_variables
    grads = tape.gradient(loss, vars)
    grads_and_vars = list(zip(grads, vars))
    self.opt.apply_gradients(grads_and_vars)

  @tf.function
  def dist_init_opt(self):
    """
    Initialized the network
    """
    self.strategy.run(self.init_opt)

  def update_model(self, x_pos, x_neg, t):
    """
    Update the model parameters in a iteration
    """
    with tf.GradientTape() as tape:
      tape.watch(self.diffusion.trainable_variables)
      loss, loss_ts, f_ts = self.diffusion.training_losses(x_pos, x_neg, t, dropout=self.hps.dropout)

    vars = self.diffusion.trainable_variables
    grads = tape.gradient(loss, vars)
    if self.hps.grad_clip:
      grads, gnorm = tf.clip_by_global_norm(grads, 1. / float(num_device()[0]))
    grads_and_vars = list(zip(grads, vars))
    grads_mean = tf.reduce_mean(tf.stack([tf.reduce_mean(tf.abs(grad)) for grad in grads], axis=0))
    grads_max = tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(grad)) for grad in grads], axis=0))
    self.opt.apply_gradients(grads_and_vars)
    self.ema.apply(self.diffusion)

    return loss, grads_mean, grads_max, loss_ts, f_ts

  @tf.function
  def train_fn(self, data):
    """
    A iteration of training
    :param data: observed clean data
    """
    x = data_preprocess(data['image'])
    _, H, W, C = x.shape
    B = self.n_per_replica
    x = tf.reshape(x, [B, H, W, C])
    if self.hps.randflip:
      x = tf.image.random_flip_left_right(x)

    t = tf.random.uniform(shape=[B], maxval=self.diffusion.num_timesteps, dtype=tf.int32)
    x_pos, x_neg = self.diffusion.q_sample_pairs(x, t)
    x_neg, disp, is_accepted = self.diffusion.p_sample_langevin(x_neg, t, dropout=self.hps.dropout)
    loss, grads_mean, grads_max, loss_ts, f_ts = self.update_model(x_pos, x_neg, t)

    return loss, grads_mean, grads_max, disp, loss_ts, f_ts, is_accepted

  @tf.function
  def distributed_train_fn(self, dist_iter):
    """
    Multi-device distributed version of train_fn
    """
    per_replica_stats = self.strategy.run(self.train_fn, args=(next(dist_iter),))
    stats = [self.strategy.reduce(tf.distribute.ReduceOp.MEAN, stat, axis=None) for stat in per_replica_stats]

    return stats

  @tf.function
  def distributed_train_fn_multisteps(self, dist_iter):
    """
    Wrap up multiple iterations within a single decoration of tf.function. Make the training faster.
    """
    disp = tf.zeros(shape=[self.hps.num_diffusion_timesteps], dtype=tf.float32)
    loss_ts = tf.zeros(shape=[self.hps.num_diffusion_timesteps], dtype=tf.float32)
    f_ts = tf.zeros(shape=[self.hps.num_diffusion_timesteps], dtype=tf.float32)
    stats = [0., 0., 0., disp, loss_ts, f_ts, 0.]
    for tt in tf.range(tf.convert_to_tensor(FLAGS.n_batch_per_iter)):
      per_replica_stats = self.strategy.run(self.train_fn, args=(next(dist_iter),))
      if tf.equal(tt, FLAGS.n_batch_per_iter - 1):
        stats = [self.strategy.reduce(tf.distribute.ReduceOp.MEAN, stat, axis=None) for stat in per_replica_stats]
    return stats

  def train(self, output_dir, output_dir_ckpt, output_dir_thread, strategy):
    self.train_setup(output_dir, output_dir_ckpt, output_dir_thread)
    self.logger.info('output dir {}'.format(self.hps.output))
    self.strategy = strategy

    # dataset
    # import resource
    # low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

    ds = datasets.get_dataset(self.hps.problem, tfds_data_dir='tensorflow_datasets')
    self.hps.img_sz = ds._img_size
    self.n_train = ds.num_train_examples
    ds = ds.train_input_fn({'batch_size': self.hps.n_batch_train})
    ds_iter = iter(ds)
    self.ds_iter = ds_iter
    self.inception_model = get_inception_model()

    x = data_preprocess(next(ds_iter)['image'])
    self.diffusion = RecoveryLikelihood(self.hps)
    self.diffusion.init(x.shape)

    lr_schedule = LambdaLr(warmup=self.hps.warmup, max_lr=self.hps.lr, total_steps=self.hps.n_iters)

    if FLAGS.opt == 'adam':
      self.opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=self.hps.beta_1)
    elif FLAGS.opt == 'adamax':
      self.opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule, beta_1=self.hps.beta_1)
    else:
      raise NotImplementedError

    # ema
    self.ema = Ema(decay=self.hps.ma_decay)
    self.diffusion_ema = RecoveryLikelihood(self.hps)
    self.diffusion_ema.init(x.shape)

    # ckpt
    i_iter_var = tf.Variable(int(0), trainable=False)
    ckpt = tf.train.Checkpoint(model=self.diffusion, model_ema=self.diffusion_ema, opt_c=self.opt,
                               i_iter_var=i_iter_var)
    if self.hps.ckpt_load:
      self.logger.info("Loading checkpoint: %s" % self.hps.ckpt_load)
      self.init_opt()
      ckpt.restore(self.hps.ckpt_load).expect_partial()

    # STATS
    stat_i = []
    stat_keys = [
      'loss',
      'fid',
      'inception_score',
      'time',
    ]
    stat = {k: [] for k in stat_keys}

    if FLAGS.eval:
      self.logger.info('========== begin evaluation =========')
      noise = tf.random.normal(shape=[64, 32, 32, 3])
      x = data_preprocess(next(ds_iter)['image'])
      x_pos_seq = self.diffusion_ema.q_sample_progressive(x)[:, :64]
      x_neg_seq, is_accepted = self.diffusion_ema.p_sample_progressive(noise)

      plot_n_by_m_steps(
        self.get_pred_by_freq(x_pos_seq), self.get_pred_by_freq(x_neg_seq),
        os.path.join(self.samples_dir, 'x_{}.png'.format(0)), n=8, m=8
      )
      self.logger.info('is_accepted={:.4f}'.format(is_accepted))
      self.eval_fid_is(full=True)

      return 0

    i_iter = i_iter_var.numpy()
    fid = 0.
    inception_score = 0.

    n_exit = 0
    start_time = time.time()
    self.logger.info('========== begin training =========')
    while i_iter < (self.hps.n_iters + 1):
      loss, grads_mean, grads_max, disp_ts, loss_ts, f_ts, is_accepted = self.train_fn(next(ds_iter))

      if i_iter % 500 == 0:
        end_time = time.time()
        start_time_next = time.time()
        disp_ts = self.get_pred_by_freq(disp_ts, last=True)
        loss_ts = self.get_pred_by_freq(loss_ts, last=True)
        f_ts = self.get_pred_by_freq(f_ts, last=True)

        disp_ts = ", ".join(["".join(str(np.around(aa, 3))) for aa in disp_ts])
        loss_ts = ", ".join(["".join(str(np.around(aa, 3))) for aa in loss_ts])
        f_ts = ", ".join(["".join(str(np.around(aa, 3))) for aa in f_ts])
        lr = self.opt._decayed_lr(tf.float32).numpy()
        self.logger.info(
          'dir={:s} i={:6d} loss={:8.4f} learning grads mean={:8.4f} grads max={:8.4f} disp={:s} loss_ts={:s} f_ts={:s} is_accepted_ts={:8.4f} lr={:4.8f} time={:.2f}s'.
            format(output_dir.split('/')[-1], i_iter, loss, grads_mean, grads_max, disp_ts, loss_ts, f_ts, is_accepted, lr, end_time - start_time)
        )
        start_time = start_time_next

      if i_iter % FLAGS.fid_n_iters == 0 and i_iter > 0:
        fid, inception_score = self.eval_fid_is()

      if i_iter % 5000 == 0 and i_iter > 0:
        self.ema.assign(model_ema=self.diffusion_ema, model=self.diffusion)
        i_iter_var.assign(i_iter)
        ckpt.write(os.path.join(self.ckpt_dir, 'ckpt-iter%d' % i_iter))

      if i_iter % 5000 == 0:
        x = data_preprocess(next(ds_iter)['image'])
        x_sample = x

        x_pos_seq = self.diffusion_ema.q_sample_progressive(x_sample)[:, :64]

        noise = tf.random.normal(shape=x_sample.shape)
        x_neg_seq = self.diffusion_ema.p_sample_progressive(noise)[0][:, :64]

        plot_n_by_m_steps(
          self.get_pred_by_freq(x_pos_seq)[:1], self.get_pred_by_freq(x_neg_seq)[:1],
          os.path.join(self.samples_dir, 'x_{}.png'.format(i_iter)), n=8, m=8
        )

        stat_i.append(i_iter)
        stat['loss'].append(loss)
        stat['fid'].append(fid)
        stat['inception_score'].append(inception_score)
        stat['time'].append(end_time - start_time)
        plot_stat(stat_keys, stat, stat_i, output_dir_thread)

      i_iter += FLAGS.n_batch_per_iter

      # set early exit
      if loss.numpy() < 0:
        n_exit += 1
      else:
        n_exit = 0
      if loss.numpy() < -1000000:
        self.logger.info('early exit due to explosion of loss')
        break
      if n_exit > 2000:
        self.logger.info('early exit due to n_exit > 2000')
      if np.isnan(loss.numpy()):
        self.logger.info('early exit due to nan')
        break

    self.logger.info('done')

  def eval_fid_is(self, full=False):
    self.logger.info('=================  computing fid  =================')
    fid_n_samples = FLAGS.fid_n_samples if not full else self.n_train
    p_samples = []
    all_logits = []
    all_pools = []
    num_batch = int(np.ceil(fid_n_samples / self.hps.fid_n_batch))
    for k in range(num_batch):
      start_time = time.time()
      noise = tf.random.normal([self.hps.fid_n_batch, self.hps.img_sz, self.hps.img_sz, 3])
      x_neg, _ = self.diffusion_ema.p_sample_progressive(noise)
      x_sample = data_postprocess(x_neg[0]).numpy()
      p_samples.append(x_sample)

      latents = run_inception_distributed(tf.convert_to_tensor(x_sample), self.inception_model)
      all_pools.append(latents["pool_3"])
      all_logits.append(latents["logits"])

      end_time = time.time()
      self.logger.info('k = {:d}, time = {:f}'.format(k, end_time - start_time))
      if full and k % 20 == 0:
        with tf.io.gfile.GFile(os.path.join(self.output_dir, 'samples.pkl'), mode='wb') as f:
          pickle.dump(p_samples, f, protocol=2)
    p_samples = np.concatenate(p_samples, axis=0)
    if full:
      with tf.io.gfile.GFile(os.path.join(self.output_dir, 'samples.pkl'), mode='wb') as f:
        pickle.dump(p_samples, f, protocol=2)

    all_logits = np.concatenate(all_logits, axis=0)[:fid_n_samples]
    all_pools = np.concatenate(all_pools, axis=0)[:fid_n_samples]

    data_stats = load_dataset_stats(self.hps)
    data_pools = data_stats["pool_3"][:fid_n_samples]
    inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
    fid = tfgan.eval.frechet_classifier_distance_from_activations(
      data_pools, all_pools)

    self.logger.info('fid p(x)={}, inception score p(x)={}'.format(fid, inception_score))

    return fid, inception_score

  def get_dist_tensor(self, noise):
    def dataset_fn(input_context):
      batch_size = input_context.get_per_replica_batch_size(noise.shape[0])
      d = tf.data.Dataset.from_tensor_slices(noise).batch(batch_size)
      return d.shard(
        input_context.num_input_pipelines,
        input_context.input_pipeline_id)

    ds = self.strategy.experimental_distribute_datasets_from_function(dataset_fn)
    noise_dist = next(iter(ds))
    return noise_dist

  def get_pred_by_freq(self, x, last=False):
    include_xpred_freq = max(1, self.hps.num_diffusion_timesteps // 10)
    idx = np.arange(self.hps.num_diffusion_timesteps // include_xpred_freq + 1) * include_xpred_freq
    if last:
      idx[-1] = idx[-1] - 1
    return tf.gather(x, idx)

  def train_setup(self, output_dir, output_dir_ckpt, output_dir_thread):
    # DIRS
    self.output_dir = output_dir_thread
    self.ckpt_dir = output_dir_ckpt + '/ckpt'
    self.ckpt_recent_dir = output_dir_ckpt + '/ckpt/recent'
    self.samples_dir = output_dir + '/samples'
    self.result_dir = os.path.join('./', 'output', 'results')

    tf.io.gfile.makedirs(self.output_dir)
    tf.io.gfile.makedirs(self.samples_dir)
    tf.io.gfile.makedirs(self.ckpt_dir)
    tf.io.gfile.makedirs(self.ckpt_recent_dir)
    tf.io.gfile.makedirs(self.result_dir)

    pygrid.copy_source(__file__, output_dir)

    # PREAMBLE
    job_id = int(self.hps.jobid)
    self.logger = pygrid.setup_logging('job{}'.format(job_id), output_dir_thread, console=False)
    self.logger.info('gpus={}'.format(self.hps.device))
    self.logger.info(self.hps)

    self.n_per_replica = self.hps.n_batch_train // num_device()[0]