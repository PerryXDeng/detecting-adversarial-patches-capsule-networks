import tensorflow as tf
import tensorflow_datasets as tfds
from config import FLAGS


def _floatify_and_normalize(datapoint):
  img = tf.cast(datapoint["image"], tf.float32) / 255
  return img, datapoint["label"]


def _train_preprocess(datapoint):
  img = datapoint["image"]
  img = tf.random_crop(img, [56, 56, 3])
  img = tf.image.random_brightness(img, max_delta=2.0)
  img = tf.image.random_contrat(img, lower=0.5, upper=1.5)
  datapoint["image"] = img
  return datapoint


def _val_preprocess(datapoint):
  img = datapoint["image"]
  img = tf.central_crop(img, [56, 56, 3])
  datapoint["image"] = img
  return datapoint


def create_inputs(is_train, force_set=None):
  # does not have test
  split = "train" if is_train else "validation"
  if force_set is not None:
    split = force_set
  data = tfds.load(name="imagenet_resized", split=split, builder_kwargs={'config':'64x64'})
  data = data.map(_floatify_and_normalize, num_parallel_calls=FLAGS.num_threads)
  if is_train:
    data = data.map(_train_preprocess, num_parallel_calls=FLAGS.num_threads)
    data = data.shuffle(2000 + 3 * FLAGS.batch_size).batch(FLAGS.batch_size, drop_remainder=True).repeat()
  else:
    data = data.map(_val_preprocess, num_parallel_calls=FLAGS.num_threads)
    data = data.batch(FLAGS.batch_size, drop_remainder=True).repeat()
  data = data.prefetch(1)
  iterator = data.make_one_shot_iterator()
  img, lab = iterator.get_next()
  output_dict = {'image': img, 'label': lab}
  return output_dict
