from datasets import fashion200k
from datasets import fashion_iq
from datasets import shoes
from datasets import vocabulary
from config import *
from model import *

import sys
import os
import numpy as np
import tensorflow as tf
import math

tf.app.flags.DEFINE_integer(
  'image_size', 224, "image size (height, width) in pixels.")
tf.app.flags.DEFINE_integer(
  'batch_size', 1, "batch size.")
tf.app.flags.DEFINE_integer(
  'num_classes', 1000, "number of classes.")
tf.app.flags.DEFINE_integer(
  'max_length', None, "maximal sequence length.")
tf.app.flags.DEFINE_integer(
  'word_embedding_size', 300, "word embedding size in input.")
tf.app.flags.DEFINE_integer(
  'text_embedding_size', 1024, "text embedding size in lstm.")
tf.app.flags.DEFINE_integer(
  'joint_embedding_size', 1024, "embedding size in the visual-semantic space.")
tf.app.flags.DEFINE_float(
  'weight_decay', 0.00004, 'weight decay on the model weights.')
tf.app.flags.DEFINE_float(
  'moving_average_decay', 0.9999, 'decay to use for the moving average.')
tf.app.flags.DEFINE_string(
  'checkpoint_dir', None, 'directory to restore the checkpoint.')
tf.app.flags.DEFINE_string(
  'exact_model_checkpoint', None, 'exact checkpoint to restore.')
tf.app.flags.DEFINE_string(
  'feature_dir', None, 'directory to extract the feature.')
tf.app.flags.DEFINE_string(
  'dataset', "fashion200k", 'either "fashion200k" or "flickr30k".')
tf.app.flags.DEFINE_string(
  'data_split', "test", 'either "train" or "test".')
tf.app.flags.DEFINE_string(
  'data_path', "/Users/yanbec/Documents/datasets/fashion200k", 'path of dataset.')
tf.app.flags.DEFINE_float(
  'text_projection_dropout', None, 'add dropout layer before projection.')
tf.app.flags.DEFINE_string(
  'image_model', "resnet_v2_ml", 'image feature extractor.')
tf.app.flags.DEFINE_string(
  'text_model', "lstm", 'text feature extractor.')
tf.app.flags.DEFINE_string(
  'image_feature_name', "global_pool", 'name of image feature.')
tf.app.flags.DEFINE_boolean(
  'is_training', False, 'whether in a training mode.')
tf.app.flags.DEFINE_boolean(
  'data_augmentation', False, 'whether uses simple image preprocessing mode.')
tf.app.flags.DEFINE_string(
  'word_embedding_dir', None, 'directory to store pretrained work_embeddings.')
tf.app.flags.DEFINE_boolean(
  'text_projection', True, 'whether to add text projection layer.')
# tf.app.flags.DEFINE_boolean(
#   'image_projection', False, 'whether to add image projection layer.')
tf.app.flags.DEFINE_boolean(
  'mod_text_projection', False, 'whether to add a text projection layer for text modification.')
tf.app.flags.DEFINE_boolean(
  'mod_text_lstm', False, 'whether to use individual lstm for text modification.')
tf.app.flags.DEFINE_boolean(
  'remove_rare_words', False, 'whether to remove the rare words.')

tf.app.flags.DEFINE_boolean(
  'query_images', True, 'whether to compute image features on all text images or the query images.')
tf.app.flags.DEFINE_boolean(
  'test_texts', False, 'whether to compute text features on all text images or the query images.')
tf.app.flags.DEFINE_string(
  'subset', None, 'can be "dress" or "shirt" or "toptee".')
tf.app.flags.DEFINE_boolean(
  'multi_scale', False, 'whether to use multi_scale features.')

FLAGS = tf.app.flags.FLAGS

squeeze = lambda x: tf.squeeze(x, [1, 2])
normalization = lambda x: tf.nn.l2_normalize(x, 1, 1e-10)
expand_dims = lambda x: tf.expand_dims(tf.expand_dims(x, 1), 1)


def _image_modify_model(source_images, modify_texts, mod_seqlengths, vocab_size):

  with tf.variable_scope(tf.get_variable_scope()):
    cnn_features_source = image_model_ml(source_images)

  lstm_features = text_model(modify_texts, mod_seqlengths, vocab_size)
  mod_text = projection_layer(lstm_features, FLAGS.text_projection_dropout, 'text')

  ### features
  composite_features = []
  for i in range(len(cnn_features_source)):
    net_name = 'attention' + str(i)
    composite_features.append(attention_layer(cnn_features_source[i], mod_text, scope=net_name))
    composite_features[i] = tf.reduce_mean(composite_features[i], [1, 2])
    composite_features[i] = normalization(composite_features[i])

  tmp = tf.concat([composite_features[0], composite_features[1]], 1)
  composite_features = tf.concat([tmp, composite_features[2]], 1)

  return composite_features


def _text_model(texts, seqlengths, vocab_size):
  with tf.variable_scope(tf.get_variable_scope()):
    lstm_features = text_model(texts, seqlengths, vocab_size)
    text_features = projection_layer(lstm_features, FLAGS.text_projection_dropout, scope='text')

  text_features = tf.nn.l2_normalize(text_features, 1, 1e-10)
  return text_features


def _image_model(images):
  with tf.variable_scope(tf.get_variable_scope()):
    cnn_features = image_model_ml(images)  # dim: 1024

  features = []
  for i in range(len(cnn_features)):
    features.append(tf.reduce_mean(cnn_features[i], [1, 2]))
    features[i] = normalization(features[i])

  tmp = tf.concat([features[0], features[1]], 1)
  features = tf.concat([tmp, features[2]], 1)
  return features


def main():
  if FLAGS.dataset == "fashion200k":
    testset = fashion200k.fashion200k(path=FLAGS.data_path, split=FLAGS.data_split)
    trainset = fashion200k.fashion200k(path=FLAGS.data_path, split="train")
  elif FLAGS.dataset == "fashion_iq":
    testset = fashion_iq.fashion_iq(path=FLAGS.data_path, split=FLAGS.data_split, subset=FLAGS.subset)
    trainset = fashion_iq.fashion_iq(path=FLAGS.data_path, split="train", subset=FLAGS.subset)
  elif FLAGS.dataset == "shoes":
    testset = shoes.shoes(path=FLAGS.data_path, split=FLAGS.data_split)
    trainset = shoes.shoes(path=FLAGS.data_path, split="train")
  else: 
    raise ValueError("dataset is unknown.")

  if FLAGS.dataset != "fashion_iq" and FLAGS.dataset != "shoes":
    ### generate source-query pairs at test time
    testset.generate_test_queries_()
  elif FLAGS.dataset == "shoes":
    testset.generate_queries_()
  else:
    testset.generate_queries_(subset=FLAGS.subset)

  vocab = vocabulary.SimpleVocab()
  all_texts = trainset.get_all_texts()

  for text in all_texts:
    vocab.add_text_to_vocab(text)
  if FLAGS.remove_rare_words:
    vocab.threshold_rare_words()
  vocab_size = vocab.get_size()

  with tf.Graph().as_default():
    if FLAGS.dataset == "shoes":
      if FLAGS.query_images:
        dataset = tf.data.Dataset.from_tensor_slices((testset.source_files, testset.modify_texts))
        num_images = len(testset.source_files)
      else:
        testset.generate_test_images_all_()
        dataset = tf.data.Dataset.from_tensor_slices((testset.database, testset.database))
        num_images = len(testset.database)
    elif FLAGS.dataset == "fashion200k":
      if FLAGS.query_images:
        dataset = tf.data.Dataset.from_tensor_slices((testset.query_filenames, testset.modify_texts))
        num_images = len(testset.test_queries)
      else:
        dataset = tf.data.Dataset.from_tensor_slices((testset.filenames, testset.texts))
        num_images = len(testset.filenames)
    else:
      testset.generate_test_images_all_(subset=FLAGS.subset)
      if FLAGS.query_images:
        dataset = tf.data.Dataset.from_tensor_slices((testset.source_files, testset.modify_texts))
        num_images = len(testset.source_files)
      else:
        dataset = tf.data.Dataset.from_tensor_slices((testset.database, testset.database))
        num_images = len(testset.database)

    dataset = dataset.prefetch(1).map(eval_image_parse_function, num_parallel_calls=1)
    data_iterator = dataset.make_one_shot_iterator()
    batch_image, batch_text = data_iterator.get_next()

    images_placeholder = tf.placeholder(tf.float32, shape=(1, FLAGS.image_size, FLAGS.image_size, 3))
    texts_placeholder = tf.placeholder(tf.int32, shape=(1, None))
    seqlengths_placeholder = tf.placeholder(tf.int32, shape=(1))

    with tf.variable_scope(tf.get_variable_scope()):
      if FLAGS.query_images:
        cnn_features = _image_modify_model(images_placeholder, texts_placeholder, seqlengths_placeholder, vocab_size)
      elif FLAGS.test_texts:
        cnn_features = _text_model(texts_placeholder, seqlengths_placeholder, vocab_size)
      else:
        cnn_features = _image_model(images_placeholder)

    if math.isnan(FLAGS.moving_average_decay):
      vars_to_restore = tf.global_variables()
      vars_to_restore = [var for var in vars_to_restore if not "ogits" in var.name]
    else:
      vars_to_restore = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay).variables_to_restore()
      vars_to_restore = {k: v for k, v in vars_to_restore.items() if not "ogits" in k}    
      
    restorer = tf.train.Saver(vars_to_restore)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth=True

    feed_dict = {
      images_placeholder: np.zeros((FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3)), 
      texts_placeholder: np.zeros((FLAGS.batch_size, 10), dtype=int),
      seqlengths_placeholder: np.zeros((FLAGS.batch_size), dtype=int)
    }

    with tf.Session(config=config) as sess:
      ### restore model
      if FLAGS.exact_model_checkpoint:
        restore_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.exact_model_checkpoint)
      else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        restore_dir = ckpt.model_checkpoint_path
      if restore_dir:
        restorer.restore(sess, restore_dir)
        global_step = restore_dir.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' % (restore_dir, global_step))
      else:
        print('No checkpoint file found')
        return

      feature_size = cnn_features.get_shape().as_list()[1]
      print('feature dim is ' + str(feature_size))

      np_image_features = np.zeros(shape=[num_images, feature_size], dtype=np.float32)
      
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
        index = 0

        print('Starting extaction on (%s). \n' % 'test data.')
        while index < num_images and not coord.should_stop():
          image_array, raw_text = sess.run([batch_image, batch_text], feed_dict=feed_dict)
          text_array = vocab.encode_text(raw_text.decode('utf-8'))
          lengths = len(text_array)

          if FLAGS.max_length is not None:
            lengths = min(FLAGS.max_length, lengths)
            max_length = FLAGS.max_length
          else:
            max_length = lengths

          feed_dict = {
            images_placeholder: image_array[np.newaxis,:,:,:], 
            texts_placeholder: np.array(text_array)[np.newaxis,:][:,0:max_length],
            seqlengths_placeholder: np.array([lengths])
          }

          np_image_feature = sess.run([cnn_features], feed_dict=feed_dict)
          np_image_features[index, :] = np_image_feature[0]
          sys.stdout.write('\r>> Extracting image features %d/%d.' % (index + 1, num_images))
          sys.stdout.flush()
          index += 1
        print('\n Finished extraction on (%s) data. \n' % FLAGS.data_split)

      except Exception as e:
        coord.request_stop(e)  
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  if not os.path.exists(FLAGS.feature_dir):
    os.mkdir(FLAGS.feature_dir)

  if FLAGS.query_images:
    filename = os.path.join(FLAGS.feature_dir, 'query_images.npy')
  elif FLAGS.test_texts:
    filename = os.path.join(FLAGS.feature_dir, 'test_texts.npy')
  else:
    filename = os.path.join(FLAGS.feature_dir, 'test_images.npy')
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  print(filename)
  np.save(filename, np_image_features) 


if __name__ == '__main__':
  main()
