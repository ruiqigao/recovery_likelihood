import math
import string

import tensorflow.compat.v1 as tf

# ===== Neural network building defaults =====
DEFAULT_DTYPE = tf.float32

def default_init(scale):
  return tf.initializers.variance_scaling(scale=1e-10 if scale == 0 else scale, mode='fan_avg', distribution='uniform')

# ===== Neural network layers =====

def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return tf.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_uppercase[:len(y.shape)])
  assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)


class nin_0(tf.keras.layers.Layer):
  def __init__(self, name, num_units, init_scale=1.):
    super(nin_0, self).__init__(name=name)
    self.num_units = num_units
    self.init_scale = init_scale

  def build(self, input_shape):
    in_dim = int(input_shape[-1])
    self.W = self.add_weight(
      'W', shape=[in_dim, self.num_units], initializer=default_init(scale=self.init_scale), dtype=DEFAULT_DTYPE
    )
    self.b = self.add_weight('b', shape=[self.num_units], initializer=tf.constant_initializer(0.), dtype=DEFAULT_DTYPE)

  def call(self, inputs, **kwargs):
    y = contract_inner(inputs, self.W) + self.b
    assert y.shape == inputs.shape[:-1] + [self.num_units]
    return y


class nin(tf.keras.layers.Layer):
  """
  network in network layer
  """
  def __init__(self, name, num_units, init_scale=1., spec_norm=False):
    super(nin, self).__init__(name=name)
    if spec_norm:
      if init_scale == 0.:
        self.nin = SN(nin_0(name, num_units, init_scale), lower_bound=True)
      else:
        self.nin = SN(nin_0(name, num_units, init_scale))
    else:
      self.nin = nin_0(name, num_units, init_scale)

  def call(self, inputs):
    return self.nin(inputs)


class SN(tf.keras.layers.Wrapper):
  """
  Spectral normalization layer
  """
  def __init__(self, layer, lower_bound=False, **kwargs):
    super(SN, self).__init__(layer, **kwargs)
    self.lower_bound = lower_bound

  def build(self, input_shape):
    if not self.layer.built:
      self.layer.build(input_shape)

      if hasattr(self.layer, 'kernel'):
        self.w = self.layer.kernel

      if hasattr(self.layer, 'W'):
        self.w = self.layer.W

      self.w_shape = self.w.shape.as_list()
      self.u = self.add_weight(shape=tuple([1, self.w_shape[-1]]),
                                 initializer=tf.compat.v1.keras.initializers.TruncatedNormal(stddev=0.02), name='sn_u',
                                 trainable=False, dtype=tf.float32,
                                 aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    super(SN, self).build(input_shape)

  def call(self, inputs, *args, **kwargs):
    self._compute_weights()
    output = self.layer(inputs, *args, **kwargs)
    return output

  def _compute_weights(self, eps=1e-12):
    w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
    _u = tf.identity(self.u)
    _v = tf.matmul(_u, tf.transpose(w_reshaped))
    _v = tf.nn.l2_normalize(_v)
    # _v = _v / tf.maximum(tf.reduce_sum(_v ** 2) ** 0.5, eps)
    _u = tf.matmul(_v, w_reshaped)
    _u = tf.nn.l2_normalize(_u)
    # _u = _u / tf.maximum(tf.reduce_sum(_u ** 2) ** 0.5, eps)

    _u = tf.stop_gradient(_u)
    _v = tf.stop_gradient(_v)
    self.u.assign(_u)
    sigma = tf.matmul(tf.matmul(_v, w_reshaped), tf.transpose(_u))

    if self.lower_bound:
      sigma = sigma + 1e-6
      w_norm = self.w / sigma * tf.minimum(sigma, 1)
    else:
      w_norm = self.w / sigma

    if hasattr(self.layer, 'kernel'):
      self.layer.kernel = w_norm

    if hasattr(self.layer, 'W'):
      self.layer.W = w_norm

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())


class dense_0(tf.keras.layers.Layer):
  def __init__(self, name, num_units, init_scale=1., bias=True):
    super(dense_0, self).__init__(name=name)
    self.num_units = num_units
    self.init_scale = init_scale
    self.bias = bias

  def build(self, input_shape):
    _, in_dim = input_shape
    self.W = self.add_weight(
      'W', shape=[in_dim, self.num_units],
      initializer=default_init(scale=self.init_scale), dtype=DEFAULT_DTYPE
    )
    if self.bias:
      self.b = self.add_weight(
        'b', shape=[self.num_units], initializer=tf.constant_initializer(0.), dtype=DEFAULT_DTYPE
      )

  def call(self, inputs, **kwargs):
    z = tf.matmul(inputs, self.W)
    if not self.bias:
      return z
    else:
      return z + self.b


class dense(tf.keras.layers.Layer):
  """
  fully-connected layer
  """
  def __init__(self, name, num_units, init_scale=1., bias=True, spec_norm=False):
    super(dense, self).__init__(name=name)
    if spec_norm:
      if init_scale == 0.:
        self.dense = SN(dense_0(name, num_units, init_scale, bias), lower_bound=True)
      else:
        self.dense = SN(dense_0(name, num_units, init_scale, bias))
    else:
      self.dense = dense_0(name, num_units, init_scale, bias)

  def call(self, inputs):
    return self.dense(inputs)


class conv2d_0(tf.keras.layers.Layer):
  def __init__(self, name, num_units, filter_size=(3, 3), stride=1, dilation=None, pad='SAME', init_scale=1., bias=True,
               use_scale=False):
    super(conv2d_0, self).__init__(name=name)
    self.num_units = num_units
    self.filter_size = filter_size
    self.stride = stride
    self.dilation = dilation
    self.pad = pad
    self.init_scale = init_scale
    self.bias = bias
    self.use_scale = use_scale

  def build(self, input_shape):
    assert input_shape.ndims == 4
    if isinstance(self.filter_size, int):
      self.filter_size = (self.filter_size, self.filter_size)
    self.W = self.add_weight('W', shape=[*self.filter_size, int(input_shape[-1]), self.num_units],
                        initializer=default_init(scale=self.init_scale), dtype=DEFAULT_DTYPE)
    if self.bias:
      self.b = self.add_weight('b', shape=[self.num_units], initializer=tf.constant_initializer(0.), dtype=DEFAULT_DTYPE)

    if self.use_scale:
      self.g = self.add_weight('g', shape=[self.num_units], initializer=tf.constant_initializer(1.), dtype=DEFAULT_DTYPE)

  def call(self, inputs):
    z = tf.nn.conv2d(inputs, self.W, strides=self.stride, padding=self.pad, dilations=self.dilation)
    if self.bias:
      z = z + self.b
    if self.use_scale:
      z = z * self.g

    return z


class conv2d(tf.keras.layers.Layer):
  """
  2d convolutional layer
  """
  def __init__(self, name, num_units, filter_size=(3, 3), stride=1, dilation=None, pad='SAME', init_scale=1., bias=True,
               spec_norm=False, use_scale=False):
    super(conv2d, self).__init__(name=name)
    if spec_norm:
      if init_scale == 0.:
        self.conv = SN(conv2d_0(name, num_units, filter_size, stride, dilation, pad, init_scale, bias, use_scale), lower_bound=True)
      else:
        self.conv = SN(conv2d_0(name, num_units, filter_size, stride, dilation, pad, init_scale, bias, use_scale))
    else:
      self.conv = conv2d_0(name, num_units, filter_size, stride, dilation, pad, init_scale, bias, use_scale)

  def call(self, inputs):
    return self.conv(inputs)


def get_timestep_embedding(timesteps, embedding_dim: int):
  """
  From Fairseq.
  Build sinusoidal embeddings.
  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".
  """
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

  half_dim = embedding_dim // 2
  emb = math.log(10000) / (half_dim - 1)
  emb = tf.exp(tf.range(half_dim, dtype=DEFAULT_DTYPE) * -emb)
  # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
  emb = tf.cast(timesteps, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
  emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
    emb = tf.pad(emb, [[0, 0], [0, 1]])
  assert emb.shape == [timesteps.shape[0], embedding_dim]
  return emb
