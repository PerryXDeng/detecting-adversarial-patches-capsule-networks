"""
License: Apache 2.0
Author: Perry Deng
E-mail: perry.deng@mail.rit.edu

Credits:
  Suofei Zhang & Hang Yu, "Matrix-Capsules-EM-Tensorflow"
  https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow
"""

import tensorflow as tf
import os, sys, time
import tensorflow.contrib.slim as slim
import datetime  # date stamp the log directory
import shutil  # to remove a directory
# to sort files in directory by date
from stat import S_ISREG, ST_CTIME, ST_MODE
import re  # for regular expressions
import sklearn.metrics as skm
import numpy as np

# Get logger that has already been created in config.py
import daiquiri

logger = daiquiri.getLogger(__name__)

# My modules
from config import FLAGS
import config as conf
import models as mod
import metrics as met
from PIL import Image


def main(args):
  # Set reproduciable random seed
  tf.set_random_seed(1234)

  # Directories
  # Get name
  split = FLAGS.load_dir.split('/')
  if split[-1]:
    name = split[-1]
  else:
    name = split[-2]

  # Get parent directory
  split = FLAGS.load_dir.split("/" + name)
  parent_dir = split[0]

  test_dir = '{}/{}/reconstructions'.format(parent_dir, name)
  test_summary_dir = test_dir + '/summary'

  # Clear the test log directory
  if (FLAGS.reset is True) and os.path.exists(test_dir):
    shutil.rmtree(test_dir)
  if not os.path.exists(test_summary_dir):
    os.makedirs(test_summary_dir)

  # Logger
  conf.setup_logger(logger_dir=test_dir, name="logger_test.txt")
  logger.info("name: " + name)
  logger.info("parent_dir: " + parent_dir)
  logger.info("test_dir: " + test_dir)

  # Load hyperparameters from train run
  conf.load_or_save_hyperparams()

  # Get dataset hyperparameters
  logger.info('Using dataset: {}'.format(FLAGS.dataset))

  # Dataset
  dataset_size_test = conf.get_dataset_size_test(FLAGS.dataset)
  num_classes = conf.get_num_classes(FLAGS.dataset)
  # train mode for random sampling
  create_inputs_test = conf.get_create_inputs(FLAGS.dataset, mode="train")

  # ----------------------------------------------------------------------------
  # GRAPH - TEST
  # ----------------------------------------------------------------------------
  logger.info('BUILD TEST GRAPH')
  g_test = tf.Graph()
  with g_test.as_default():
    tf.train.get_or_create_global_step()
    # Get data
    input_dict = create_inputs_test()
    batch_x = input_dict['image']
    batch_labels = input_dict['label']

    # Build architecture
    build_arch = conf.get_dataset_architecture(FLAGS.dataset)
    # for baseline
    # build_arch = conf.get_dataset_architecture('baseline')

    # --------------------------------------------------------------------------
    # MULTI GPU - TEST
    # --------------------------------------------------------------------------
    # Calculate the logits for each model tower
    with tf.device('/gpu:0'):
      with tf.name_scope('tower_0') as scope:
        with slim.arg_scope([slim.variable], device='/cpu:0'):
          loss, logits, recon, cf_recon = tower_fn(
            build_arch,
            batch_x,
            batch_labels,
            scope,
            num_classes,
            reuse_variables=tf.AUTO_REUSE,
            is_train=False)

        # Keep track of losses and logits across for each tower
        recon_images = tf.reshape(recon, batch_x.get_shape())
        cf_recon_images = tf.reshape(cf_recon, batch_x.get_shape())
        images = {"reconstructed_images":recon_images, "reconstructed_images_zeroed_background":cf_recon_images}
    saver = tf.train.Saver(max_to_keep=None)


    # --------------------------------------------------------------------------
    # SESSION - TEST
    # --------------------------------------------------------------------------
    # sess_test = tf.Session(
    #    config=tf.ConfigProto(allow_soft_placement=True,
    #                          log_device_placement=False),
    #    graph=g_test)
    # Perry: added in for RTX 2070 incompatibility workaround
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess_test = tf.Session(config=config, graph=g_test)

    # sess_test.run(tf.local_variables_initializer())
    # sess_test.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(
      test_summary_dir,
      graph=sess_test.graph)

    ckpts_to_test = []
    load_dir_chechpoint = os.path.join(FLAGS.load_dir, "train", "checkpoint")

    # Evaluate the latest ckpt in dir
    if FLAGS.ckpt_name is None:
      latest_ckpt = tf.train.latest_checkpoint(load_dir_chechpoint)
      ckpts_to_test.append(latest_ckpt)
    # Evaluate all ckpts in dir
    else:
      ckpt_name = os.path.join(load_dir_chechpoint, FLAGS.ckpt_name)
      ckpts_to_test.append(ckpt_name)

      # --------------------------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------------------------
    # Run testing on checkpoints
    for ckpt in ckpts_to_test:
      saver.restore(sess_test, ckpt)

      for i in range(dataset_size_test):
        out = sess_test.run([images])
        reconstructed_image, reconstructed_image_zeroed_background =\
            out[0]["reconstructed_images"], out[0]["reconstructed_images_zeroed_background"]
        if reconstructed_image.shape[0] == 1:
          reconstructed_image = np.squeeze(reconstructed_image, axis=0)
          reconstructed_image_zeroed_background = np.squeeze(reconstructed_image_zeroed_background, axis=0)
        if reconstructed_image.shape[-1] == 1:
          reconstructed_image = np.squeeze(reconstructed_image, axis=-1)
          reconstructed_image_zeroed_background = np.squeeze(reconstructed_image_zeroed_background, axis=-1)
        reconstructed_image = Image.fromarray((reconstructed_image * 255).astype('uint8'))
        reconstructed_image_zeroed_background = Image.fromarray((reconstructed_image_zeroed_background * 255).astype('uint8'))
        reconstructed_image.show()
        reconstructed_image_zeroed_background.show()


def tower_fn(build_arch,
             x,
             y,
             scope,
             num_classes,
             is_train=False,
             reuse_variables=None):
  """Model tower to be run on each GPU.

  Author:
    Ashley Gritzman 27/11/2018

  Args:
    build_arch:
    x: split of batch_x allocated to particular GPU
    y: split of batch_y allocated to particular GPU
    scope:
    num_classes:
    is_train:
    reuse_variables: False for the first GPU, and True for subsequent GPUs

  Returns:
    loss: mean loss across samples for one tower (scalar)
    scores:
      If the architecture is a capsule network, then the scores are the output
      activations of the class caps.
      If the architecture is the CNN baseline, then the scores are the logits of
      the final layer.
      (samples_per_tower, n_classes)
      (64/4=16, 5)
  """

  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    output = build_arch(x, is_train, num_classes=num_classes)
  loss = mod.total_loss(output, y)
  return loss, output['scores'], output['decoder_out'], output['zeroed_bg_decoder_out']

if __name__ == "__main__":
  tf.app.run()
