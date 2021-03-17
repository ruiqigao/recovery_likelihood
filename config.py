"""
Configuration of models, dataset and training
"""

from absl import flags

# utils
flags.DEFINE_integer('jobid', 0, 'job id')
flags.DEFINE_string('logdir', '', 'directory of logging')
flags.DEFINE_boolean('eager', False, 'open the eager mode or not')
flags.DEFINE_string('ckpt_load', None, 'checkpoint file to load')
flags.DEFINE_string('device', '0', 'device id')
flags.DEFINE_boolean('tpu', False, 'use TPU or not')
flags.DEFINE_string('tpu_name', None, 'TPU name')
flags.DEFINE_string('tpu_zone', None, 'TPU zone')
flags.DEFINE_integer('rnd_seed', 1, 'random seed')

# dataset
flags.DEFINE_string('problem', 'cifar10',
                    'svhn / cifar10 / celeba / lsun_church128 / lsun_bedroom128 / lsun_church64 / lsun_bedroom64')

# train
flags.DEFINE_integer('n_batch_train', 64, 'batch size in training')
flags.DEFINE_float('lr', 1e-4, 'learning rate')
flags.DEFINE_float('beta_1', 0.9, 'beta1 in adam optimizer')
flags.DEFINE_integer('n_iters', 1000000, 'number of training iterations')
flags.DEFINE_boolean('grad_clip', False, 'norm clip the gradient')
flags.DEFINE_integer('warmup', 1000, 'number of warm-up iterations')
flags.DEFINE_integer('n_batch_per_iter', 1, 'number of iterations to wrap up in a single decoration') # TODO: change back to 50
flags.DEFINE_boolean('cosine_decay', False, 'cosine decay the learning rate')
flags.DEFINE_string('opt', 'adam', 'adam / adamax')

# eval
flags.DEFINE_boolean('eval', False, 'evaluation mode')
flags.DEFINE_integer('include_xpred_freq', 1, 'Per timesteps to output images')
flags.DEFINE_boolean('eval_fid', False, 'compute FID/IS scores in evaluation mode')
flags.DEFINE_integer('fid_n_samples', 64, 'number of samples to use to compute the FID/IS scores')  # TODO 50000
flags.DEFINE_integer('fid_n_iters', 40000, 'number of every iterations to computer FID/IS scores')
flags.DEFINE_integer('fid_n_batch', 64, 'batch size to compute FID/IS scores')  # TODO change back to 2560

# model
flags.DEFINE_integer('num_res_blocks', 2, 'number of residual blocks')
flags.DEFINE_integer('num_diffusion_timesteps', 6, 'number of time steps')
flags.DEFINE_boolean('randflip', True, 'random flip images')
flags.DEFINE_float('dropout', 0., 'dropout in the residual blocks')
flags.DEFINE_string('normalize', None, 'None / batch_norm / group_norm / instance_norm')
flags.DEFINE_string('act', 'lrelu', 'lrelu / swish')
flags.DEFINE_string('final_act', 'relu', 'relu / lrelu / swish')
flags.DEFINE_boolean('use_attention', False, 'add attention layers in residual blocks')
flags.DEFINE_boolean('resamp_with_conv', False, 'downsample / upsample with conv layers')
flags.DEFINE_boolean('spec_norm', True, 'spectral normalization')
flags.DEFINE_boolean('res_conv_shortcut', True, 'use conv shortcut layers in residual blocks')
flags.DEFINE_boolean('res_use_scale', True, 'learn scaling parameters in the 2nd conv layer of each res block')
flags.DEFINE_float('ma_decay', 0.999, 'exp moving average for testing [0.999]')
flags.DEFINE_float('noise_scale', 1.0, 'MCMC sampling noise scale, 1.0 in training / 0.99 in testing')

# sampling
flags.DEFINE_integer('mcmc_num_steps', 30, 'number of mcmc sampling steps')
flags.DEFINE_float('mcmc_step_size_b_square', 2e-4, 'scaling parameter of step sizes')

FLAGS = flags.FLAGS
