"""Dataset loading utilities.

All images are scaled to [0, 255] instead of [0, 1]
"""

import functools
import tensorflow as tf
import tensorflow_datasets as tfds


def pack(image, label):
  label = tf.cast(label, tf.int32)
  return {'image': image, 'label': label}


class SimpleDataset:
  DATASET_NAMES = ('cifar10', 'celebahq128', 'celebahq256', 'svhn', 'mnist', 'celeba', 'lsun_church64',
                   'lsun_bedroom64', 'lsun_bedroom128', 'lsun_church128', 'cifar100')

  def __init__(self, name, tfds_data_dir):
    self._name = name
    self._data_dir = tfds_data_dir
    self._img_size = {'svhn': 32, 'mnist': 28, 'cifar10': 32, 'cifar100': 32, 'celebahq128': 128, 'celebahq256': 256, 'celeba': 32,
                      'lsun_church64': 64, 'lsun_bedroom64': 64, 'lsun_church128': 128, 'lsun_bedroom128': 128}[name]
    if name == 'mnist':
      self._img_shape = [self._img_size, self._img_size, 1]
    else:
      self._img_shape = [self._img_size, self._img_size, 3]
    self._tfds_name = {
      'svhn': 'svhn_cropped:3.0.0',
      'cifar10': 'cifar10:3.0.2',
      'cifar100': 'cifar100:3.0.2',
      'celebahq128': 'celeb_a_hq/128',
      'celebahq256': 'celeb_a_hq/256:2.0.0',
      'mnist': 'mnist:3.0.1',
      'celeba': 'celeb_a',
      'lsun_church64': 'lsun/church_outdoor',
      'lsun_church128': 'lsun/church_outdoor',
      'lsun_bedroom64': 'lsun/bedroom',
      'lsun_bedroom128': 'lsun/bedroom'
    }[name]
    self.num_train_examples, self.num_eval_examples = {
      'svhn': (73257, 26032),
      'cifar10': (50000, 10000),
      'cifar100': (50000, 10000),
      'celebahq128': (30000, 0),
      'celebahq256': (30000, 0),
      'mnist': (60000, 10000),
      'celeba': (162770, 0),
      'lsun_church64': (126227, 300),
      'lsun_church128': (126227, 300),
      'lsun_bedroom64': (3033042, 300),
      'lsun_bedroom128': (3033042, 300),
    }[name]
    self.num_classes = 1  # unconditional
    self.eval_split_name = {
      'svhn': 'test',
      'cifar10': 'test',
      'cifar100': 'test',
      'celebahq128': None,
      'celebahq256': None,
      'mnist': 'test',
      'celeba': None,
      'lsun_church64': None,
      'lsun_church128': None,
      'lsun_bedroom64': None,
      'lsun_bedroom128': None,
    }[name]

  @property
  def image_shape(self):
    """Returns a tuple with the image shape."""
    return tuple(self._img_shape)

  def _proc_and_batch(self, ds, batch_size):
    def _process_data(x_):
      img_ = tf.cast(x_['image'], tf.int32)
      if self._name == 'celeba':
        img_ = tf.image.resize(img_[20: -20], [self._img_size, self._img_size], antialias=True)
      elif self._name == 'lsun_church64' or self._name == 'lsun_bedroom64' or self._name == 'lsun_church128' or self._name == 'lsun_bedroom128':
        crop = tf.minimum(tf.shape(img_)[0], tf.shape(img_)[1])
        img_ = img_[(tf.shape(img_)[0] - crop) // 2 : (tf.shape(img_)[0] + crop) // 2, (tf.shape(img_)[1] - crop) // 2 : (tf.shape(img_)[1] + crop) // 2]
        img_ = tf.image.resize(img_, [self._img_size, self._img_size], antialias=True)
      img_.set_shape(self._img_shape)
      return pack(image=img_, label=tf.constant(0, dtype=tf.int32))

    ds = ds.map(_process_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def train_input_fn(self, params):
    ds = tfds.load(self._tfds_name, split='train', shuffle_files=True, data_dir=self._data_dir, try_gcs=True)
    ds = ds.repeat()
    ds = ds.shuffle(50000)
    return self._proc_and_batch(ds, params['batch_size'])

  def train_one_pass_input_fn(self, params):
    ds = tfds.load(self._tfds_name, split='train', shuffle_files=False, data_dir=self._data_dir, try_gcs=True)
    return self._proc_and_batch(ds, params['batch_size'])

  def eval_input_fn(self, params):
    if self.eval_split_name is None:
      return None
    ds = tfds.load(self._tfds_name, split=self.eval_split_name, shuffle_files=False, data_dir=self._data_dir, try_gcs=True)
    return self._proc_and_batch(ds, params['batch_size'])


DATASETS = {
  "mnist": functools.partial(SimpleDataset, name="mnist"),
  "svhn": functools.partial(SimpleDataset, name="svhn"),
  "cifar10": functools.partial(SimpleDataset, name="cifar10"),
  "cifar100": functools.partial(SimpleDataset, name="cifar100"),
  "celebahq128": functools.partial(SimpleDataset, name="celebahq128"),
  "celebahq256": functools.partial(SimpleDataset, name="celebahq256"),
  "celeba": functools.partial(SimpleDataset, name="celeba"),
  "lsun_bedroom64": functools.partial(SimpleDataset, name="lsun_bedroom64"),
  "lsun_bedroom128": functools.partial(SimpleDataset, name="lsun_bedroom128"),
  "lsun_church64": functools.partial(SimpleDataset, name="lsun_church64"),
  "lsun_church128": functools.partial(SimpleDataset, name="lsun_church128"),
}


def get_dataset(name, *, tfds_data_dir=None, seed=547):
  """
  Instantiates a data set and sets the random seed.
  """

  kwargs = {}
  kwargs['tfds_data_dir'] = tfds_data_dir
  name_prefix = name

  if name_prefix not in ['lsun', *SimpleDataset.DATASET_NAMES]:
    kwargs['seed'] = seed

  if name_prefix not in DATASETS:
    raise ValueError("Dataset %s is not available." % name)

  return DATASETS[name_prefix](**kwargs)


def data_preprocess(x):
  x = tf.cast(x, tf.float32)
  x = x / 127.5 - 1.
  return x


def data_postprocess(x):
  return tf.cast(tf.clip_by_value((x + 1.) * 127.5, 0., 255.), tf.uint8)