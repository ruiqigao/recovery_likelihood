import tensorflow_probability as tfp
tfd = tfp.distributions
import pygrid as pygrid
from absl import app
import tensorflow.compat.v2 as tf
from train_utils import *
from train import Trainer
from train_distributed import Trainer as Trainer_dist


def main(argv):
  del argv
  LARGE_DATASETS = ["celebahq128", "lsun_bedroom128", "lsun_bedroom64", 'lsun_church128', 'lsun_church64', 'celeba']
  exp_id = pygrid.get_exp_id(__file__)
  output_dir = pygrid.get_output_dir(exp_id, './')
  if FLAGS.problem in LARGE_DATASETS:
    FLAGS.fid_n_samples = 2560
    FLAGS.fid_n_batch = 640
  elif FLAGS.problem == 'celebahq256':
    FLAGS.fid_n_samples = 1280
    FLAGS.fid_n_batch = 160

  hps = AttrDict(get_flag_dict())
  hps.output = output_dir

  if hps.device:
      set_gpu(hps.device)
  init_tf2(tf_eager=hps.eager, tf_memory_growth=True)

  if hps.tpu:
    resolver = setup_tpu()
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    model = Trainer_dist(hps=hps)
  else:
    strategy = None
    model = Trainer(hps=hps)

  set_seed(hps.rnd_seed)
  model.train(output_dir, output_dir, output_dir, strategy)


if __name__ == '__main__':
  app.run(main)