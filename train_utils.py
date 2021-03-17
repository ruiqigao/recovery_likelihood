"""
Training utilities
"""
import tensorflow.compat.v2 as tf
from config import FLAGS
from PIL import Image
import math
from datasets import data_preprocess, data_postprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from logging import StreamHandler
import logging
import random


def cosine_decay(lr, step, total_steps):
  ratio = tf.maximum(0., step / total_steps)
  mult = 0.5 * (1. + tf.cos(np.pi * ratio))
  return mult * lr


def get_warmed_up_lr(step, max_lr, n_warmup, total_steps):
  step = tf.cast(step, tf.float32)
  if FLAGS.cosine_decay:
    lr = cosine_decay(max_lr, tf.minimum(step - n_warmup, total_steps - n_warmup), total_steps - n_warmup)
  else:
    lr = max_lr
  warmup = tf.minimum(1., step / n_warmup)

  return lr * warmup


class LambdaLr(tf.optimizers.schedules.LearningRateSchedule):
  def __init__(self, *, max_lr, warmup, total_steps):
    super(LambdaLr, self).__init__()
    self.max_lr = max_lr
    self.warmup = warmup
    self.total_steps = total_steps

  def __call__(self, step):
    return get_warmed_up_lr(step, self.max_lr, self.warmup, self.total_steps)

  def get_config(self):
    return {}


class Ema():
  def __init__(self, decay):
    self.ema = tf.train.ExponentialMovingAverage(decay=decay)

  def get_ordered_values(self, vars, order):
    return [vars[v] for v in order]

  def get_vars(self, model_vars):
    return [v.numpy() for v in self.get_ordered_values(self.ema._averages, model_vars)]

  def load(self, model_vars, model_ema_vars):
    # ema
    self.ema.apply(model_vars)
    ordered_ema_trg = self.get_ordered_values(self.ema._averages, model_vars)
    ordered_ema_src = model_ema_vars
    for (v1, v2) in zip(ordered_ema_src, ordered_ema_trg):
      assert v1.shape == v2.shape
      v2.assign(v1.read_value())

  def apply(self, model):
    vars = model.variables
    self.ema.apply(vars)

  def assign(self, model_ema, model):
    vars_trg, vars_src = model_ema.variables, model.variables
    for v1, v2 in zip(vars_trg, vars_src):
      v1.assign(self.ema.average(v2).read_value())


def num_device():
  device_type = 'TPU'
  num_devices = len(tf.config.list_logical_devices('TPU'))
  if num_devices == 0:
    num_devices = len(tf.config.list_logical_devices('GPU'))
    device_type = 'GPU'
  return num_devices, device_type


def to_grid(image_batch, size, edge=0):
  h, w = image_batch.shape[1], image_batch.shape[2]
  c = image_batch.shape[3]
  img = np.ones((int(h * size[0]) + edge * (size[0] - 1), w * size[1] + edge * (size[1] - 1), c)) * 255
  for idx, im in enumerate(image_batch):
    i = idx % size[1]
    j = idx // size[1]
    img[j * (h + edge):j * (h + edge) + h, i * (w + edge):i * (w + edge) + w, :] = im
  return img


def to_grid_n_batch(image_n_batch, size, edge=0):
  img = []
  for i in range(len(image_n_batch)):
    img.append(to_grid(image_n_batch[i], size, edge))

  return np.stack(img, axis=0)


def plot(x, fp, n):
  with tf.io.gfile.GFile(fp, mode='w') as f:
    assert int(math.sqrt(n)) ** 2 == n
    Image.fromarray(np.squeeze(to_grid(x, [int(math.sqrt(n)), int(math.sqrt(n))]).astype(np.uint8))).save(f)


def plot_n_by_m(x, fp, n, m):
  with tf.io.gfile.GFile(fp, mode='w') as f:
    Image.fromarray(np.squeeze(to_grid(data_postprocess(x), [int(n), int(m)], edge=2).astype(np.uint8))).save(f)


def plot_n_by_m_steps(x_true, x_pred, fp, n, m):
  assert x_true.shape == x_pred.shape
  x_true = to_grid_n_batch(data_postprocess(x_true), [int(n), int(m)])
  x_true = to_grid(x_true, [1, len(x_true)], edge=5)
  x_pred = to_grid_n_batch(data_postprocess(x_pred), [int(n), int(m)])
  x_pred = to_grid(x_pred, [1, len(x_pred)], edge=5)

  img = to_grid(np.stack([x_true, x_pred], axis=0), [2, 1], edge=5).astype(np.uint8)
  if img.shape[-1] == 1:
    img = np.tile(img, [1, 1, 3])
  with tf.io.gfile.GFile(fp, mode='w') as f:
    Image.fromarray(img).save(f)


def plot_stat(stat_keys, stats, stats_i, output_dir):
  from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
  p_n = len(stats)
  fig = plt.figure(figsize=(20, p_n * 5))
  canvas = FigureCanvas(fig)

  p_i = 1
  for k in stat_keys:
    plt.subplot(p_n, 1, p_i)
    plt.plot(stats_i, stats[k])
    if k == 'fid' or k == 'inception_score':
      for i, txt in enumerate(stats[k]):
        if stats_i[i] % FLAGS.fid_n_iters == 0:
          plt.annotate(str(np.round(txt, decimals=2)), (stats_i[i], stats[k][i]))
    plt.ylabel(k)
    p_i += 1
  canvas.draw()
  width, height = fig.get_size_inches() * fig.get_dpi()
  img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

  with tf.io.gfile.GFile(os.path.join(output_dir, 'stat.png'), mode='w') as f:
    Image.fromarray(img).save(f)
  plt.close()


def set_gpu(gpus='0'):
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)


def setup_tpu():
  """setup tpu."""
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu_name, zone=FLAGS.tpu_zone)
  tf.config.experimental_connect_to_cluster(resolver)
  topology = tf.tpu.experimental.initialize_tpu_system(resolver)
  logging.info('topology.mesh_shape: %s', topology.mesh_shape)
  logging.info('topology._device_coordinates: %s', topology.device_coordinates)
  return resolver



def init_tf2(tf_eager, tf_memory_growth=True):
  tf.enable_v2_behavior()
  tf.config.set_soft_device_placement(True)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if tf_memory_growth and gpus:
    for gpu in gpus:
      # rtx needs memory growth for multi-gpu
      # https://github.com/tensorflow/tensorflow/issues/29632
      tf.config.experimental.set_memory_growth(gpu, True)

  tf.config.experimental_run_functions_eagerly(tf_eager)
  tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})


####### logging ########
def set_seed(seed):
  assert seed
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  tf.random.set_seed(seed)


class FileHandler(StreamHandler):
  def __init__(self, f, mode='a', encoding=None, delay=False):
    self.f = f
    self.mode = mode
    self.encoding = encoding
    self.delay = delay
    StreamHandler.__init__(self, f)

  def close(self):
    self.acquire()
    try:
      try:
        if self.stream:
          try:
            self.flush()
          finally:
            stream = self.stream
            self.stream = None
            if hasattr(stream, "close"):
              stream.close()
      finally:
        StreamHandler.close(self)
    finally:
      self.release()

  def emit(self, record):
    if self.stream is None:
      self.stream = self._open()
    StreamHandler.emit(self, record)

  def __repr__(self):
    level = 'info'
    return '<%s %s (%s)>' % (self.__class__.__name__, self.baseFilename, level)


def setup_logging(name, f, console=True):
  log_format = logging.Formatter("%(asctime)s : %(message)s")
  logger = logging.getLogger(name)
  logger.handlers = []
  file_handler = FileHandler(f)
  file_handler.setFormatter(log_format)
  logger.addHandler(file_handler)
  if console:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
  logger.setLevel(logging.INFO)
  return logger


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


def get_flag_dict():
  d = {}
  for (k, v) in FLAGS.__flags.items():
    d[k] = FLAGS[k]._value
  return d
