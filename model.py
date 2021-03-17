"""
Model class, including generating diffused samples, training loss and sampling. Adapted from DDPM code by Jonathan Ho.
"""

import numpy as np
import tensorflow.compat.v2 as tf
from config import FLAGS
from network import net_res_temb2


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
  betas = np.append(betas, 1.)
  assert betas.shape == (num_diffusion_timesteps + 1,)


  return betas


def get_sigma_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
  """
  Get the noise level schedule
  :param beta_start: begin noise level
  :param beta_end: end noise level
  :param num_diffusion_timesteps: number of timesteps
  :return:
  -- sigmas: sigma_{t+1}, scaling parameter of epsilon_{t+1}
  -- a_s: sqrt(1 - sigma_{t+1}^2), scaling parameter of x_t
  """
  betas = np.linspace(beta_start, beta_end, 1000, dtype=np.float64)
  betas = np.append(betas, 1.)
  assert isinstance(betas, np.ndarray)
  betas = betas.astype(np.float64)
  assert (betas > 0).all() and (betas <= 1).all()
  sqrt_alphas = np.sqrt(1. - betas)
  idx = tf.cast(np.concatenate([np.arange(num_diffusion_timesteps) * (1000 // ((num_diffusion_timesteps - 1) * 2)), [999]]), dtype=tf.int32)
  a_s = np.concatenate(
    [[np.prod(sqrt_alphas[: idx[0] + 1])],
     np.asarray([np.prod(sqrt_alphas[idx[i - 1] + 1: idx[i] + 1]) for i in np.arange(1, len(idx))])])
  sigmas = np.sqrt(1 - a_s ** 2)

  return sigmas, a_s


class RecoveryLikelihood(tf.keras.Model):
  def __init__(self, hps):
    super(RecoveryLikelihood, self).__init__()
    self.hps = hps
    self.num_timesteps = FLAGS.num_diffusion_timesteps

    self.sigmas, self.a_s = get_sigma_schedule(beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=self.num_timesteps)
    self.a_s_cum = np.cumprod(self.a_s)
    self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
    self.a_s_prev = self.a_s.copy()
    self.a_s_prev[-1] = 1
    self.is_recovery = np.ones(self.num_timesteps + 1, dtype=np.float32)
    self.is_recovery[-1] = 0

    if self.hps.img_sz == 32:
      ch_mult = (1, 2, 2, 2)
    elif self.hps.img_sz == 128:
      ch_mult = (1, 2, 2, 2, 4, 4)
    elif self.hps.img_sz == 64:
      ch_mult = (1, 2, 2, 2, 4)
    elif self.hps.img_sz == 256:
      ch_mult = (1, 1, 2, 2, 2, 4, 4,)
    else:
      raise NotImplementedError

    self.net = net_res_temb2(name='net', ch=128, ch_mult=ch_mult, num_res_blocks=FLAGS.num_res_blocks, attn_resolutions=(16,))

  def init(self, x_shape):
    """
    Initialization function to activate model weights.
    :param x_shape: input date shape
    """
    x = tf.random.uniform(x_shape, minval=-.5, maxval=.5)
    self.net(x, 0, dropout=0.)

  @staticmethod
  def _extract(a, t, x_shape):
    """
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    if isinstance(t, int) or len(t.shape) == 0:
      t = tf.ones(x_shape[0], dtype=tf.int32) * t
    bs, = t.shape
    assert x_shape[0] == bs
    out = tf.gather(tf.convert_to_tensor(a, dtype=tf.float32), t)
    assert out.shape == [bs]
    return tf.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

  def q_sample(self, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for 1 step)
    """
    if noise is None:
      noise = tf.random.normal(shape=x_start.shape)
    assert noise.shape == x_start.shape
    x_t = self._extract(self.a_s_cum, t, x_start.shape) * x_start + \
          self._extract(self.sigmas_cum, t, x_start.shape) * noise

    return x_t

  def q_sample_pairs(self, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = tf.random.normal(shape=x_start.shape)
    x_t = self.q_sample(x_start, t)
    x_t_plus_one = self._extract(self.a_s, t+1, x_start.shape) * x_t + \
                   self._extract(self.sigmas, t+1, x_start.shape) * noise

    return x_t, x_t_plus_one

  def q_sample_progressive(self, x_0):
    """
    Generate a full sequence of disturbed images
    """
    x_preds = []
    for t in range(self.num_timesteps + 1):
      t_now = tf.ones([x_0.shape[0]], dtype=tf.int32) * t
      x = self.q_sample(x_0, t_now)
      x_preds.append(x)
    x_preds = tf.stack(x_preds, axis=0)

    return x_preds

  # === Training loss ===
  def training_losses(self, x_pos, x_neg, t, *, dropout=0.):
    """
    Training loss calculation
    """
    a_s = self._extract(self.a_s_prev, t + 1, x_pos.shape)
    y_pos = a_s * x_pos
    y_neg = a_s * x_neg
    pos_f = self.net(y_pos, t, dropout=dropout)
    neg_f = self.net(y_neg, t, dropout=dropout)
    loss = - (pos_f - neg_f)

    loss_scale = 1.0 / (tf.cast(tf.gather(self.sigmas, t + 1), tf.float32) / self.sigmas[1])
    loss = loss_scale * loss

    loss_ts = tf.math.unsorted_segment_mean(tf.abs(loss), t, self.num_timesteps)
    f_ts = tf.math.unsorted_segment_mean(tf.abs(pos_f), t, self.num_timesteps)

    return tf.nn.compute_average_loss(loss, global_batch_size=self.hps.n_batch_train), loss_ts, f_ts

  def log_prob(self, y, t, tilde_x, b0, sigma, is_recovery, *, dropout):
    return self.net(y, t, dropout=dropout) / tf.reshape(b0, [-1]) - tf.reduce_sum((y - tilde_x) ** 2 / 2 / sigma ** 2 * is_recovery, axis=[1, 2, 3])

  def grad_f(self, y, t, tilde_x, b0, sigma, is_recovery, *, dropout):
    with tf.GradientTape() as tape:
      tape.watch(y)
      log_p_y = self.log_prob(y, t, tilde_x, b0, sigma, is_recovery, dropout=dropout)
    grad_y = tape.gradient(log_p_y, y)
    return grad_y, log_p_y

  # === Sampling ===
  def p_sample_langevin(self, tilde_x, t, *, dropout):
    """
    Langevin sampling function
    """
    sigma = self._extract(self.sigmas, t + 1, tilde_x.shape)
    sigma_cum = self._extract(self.sigmas_cum, t, tilde_x.shape)
    is_recovery = self._extract(self.is_recovery, t + 1, tilde_x.shape)
    a_s = self._extract(self.a_s_prev, t + 1, tilde_x.shape)

    c_t_square = sigma_cum / self.sigmas_cum[0]
    step_size_square = c_t_square * self.hps.mcmc_step_size_b_square * sigma ** 2

    y = tf.identity(tilde_x)
    is_accepted_summary = tf.zeros(y.shape[0], dtype=tf.float32)
    grad_y, log_p_y = self.grad_f(y, t, tilde_x, step_size_square, sigma, is_recovery, dropout=dropout)

    for _ in tf.range(tf.convert_to_tensor(self.hps.mcmc_num_steps)):
      noise = tf.random.normal(y.shape)
      y_new = y + 0.5 * step_size_square * grad_y + tf.sqrt(step_size_square) * noise * FLAGS.noise_scale

      grad_y_new, log_p_y_new = self.grad_f(y_new, t, tilde_x, step_size_square, sigma, is_recovery, dropout=dropout)
      y, grad_y, log_p_y = y_new, grad_y_new, log_p_y_new

    is_accepted_summary = is_accepted_summary / tf.convert_to_tensor(self.hps.mcmc_num_steps, dtype=tf.float32)
    is_accepted_summary = tf.reduce_mean(is_accepted_summary)

    x = y / a_s

    disp = tf.math.unsorted_segment_mean(
      tf.norm(tf.reshape(x, [x.shape[0], -1]) - tf.reshape(tilde_x, [tilde_x.shape[0], -1]), axis=1),
      t, self.num_timesteps)

    return x, disp, is_accepted_summary

  @tf.function
  def p_sample_progressive(self, noise):
    """
    Sample a sequence of images with the sequence of noise levels
    """
    num = noise.shape[0]
    x_neg_t = noise
    x_neg = tf.zeros([self.hps.num_diffusion_timesteps, num, self.hps.img_sz, self.hps.img_sz, 3], dtype=tf.float32)
    x_neg = tf.concat([x_neg, tf.expand_dims(noise, axis=0)], axis=0)
    is_accepted_summary = tf.constant(0.)

    for t in tf.range(self.hps.num_diffusion_timesteps - 1, -1, -1):
      x_neg_t, _, is_accepted = self.p_sample_langevin(x_neg_t, t, dropout=0.)
      is_accepted_summary = is_accepted_summary + is_accepted
      x_neg_t = tf.reshape(x_neg_t, [num, self.hps.img_sz, self.hps.img_sz, 3])
      insert_mask = tf.equal(t, tf.range(self.hps.num_diffusion_timesteps + 1, dtype=tf.int32))
      insert_mask = tf.reshape(tf.cast(insert_mask, dtype=tf.float32), [-1, *([1] * len(noise.shape))])
      x_neg = insert_mask * tf.expand_dims(x_neg_t, axis=0) + (1. - insert_mask) * x_neg
    is_accepted_summary = is_accepted_summary / tf.convert_to_tensor(self.hps.num_diffusion_timesteps, dtype=tf.float32)
    return x_neg, is_accepted_summary

  def p_sample_progressive_inner(self, noise):
    """
    Sample a sequence of images with the sequence of noise levels, without tf.function decoration
    """
    num = noise.shape[0]
    x_neg_t = noise
    x_neg = tf.zeros([self.hps.num_diffusion_timesteps, num, self.hps.img_sz, self.hps.img_sz, 3], dtype=tf.float32)
    x_neg = tf.concat([x_neg, tf.expand_dims(noise, axis=0)], axis=0)
    is_accepted_summary = tf.constant(0.)

    for t in tf.range(self.hps.num_diffusion_timesteps - 1, -1, -1):
      x_neg_t, _, is_accepted = self.p_sample_langevin(x_neg_t, t, dropout=0.)
      is_accepted_summary = is_accepted_summary + is_accepted
      x_neg_t = tf.reshape(x_neg_t, [num, self.hps.img_sz, self.hps.img_sz, 3])
      insert_mask = tf.equal(t, tf.range(self.hps.num_diffusion_timesteps + 1, dtype=tf.int32))
      insert_mask = tf.reshape(tf.cast(insert_mask, dtype=tf.float32), [-1, *([1] * len(noise.shape))])
      x_neg = insert_mask * tf.expand_dims(x_neg_t, axis=0) + (1. - insert_mask) * x_neg
    is_accepted_summary = is_accepted_summary / tf.convert_to_tensor(self.hps.num_diffusion_timesteps, dtype=tf.float32)
    return x_neg, is_accepted_summary

  @tf.function
  def distribute_p_sample_progressive(self, noise, strategy):
    """
    Multi-device distributed version of p_sample_progressive
    """
    samples, is_accepted = strategy.run(self.p_sample_progressive_inner, args=(noise,))
    samples = tf.concat(samples.values, axis=1)
    is_accepted = strategy.reduce(tf.distribute.ReduceOp.MEAN, is_accepted, axis=None)

    return samples, is_accepted