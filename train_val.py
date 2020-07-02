from datasets import fashion_iq
from datasets import shoes
from datasets import vocabulary
from config import *
from model import *

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import batch_and_drop_remainder

##############################################################################################
# Govern training 
##############################################################################################
tf.app.flags.DEFINE_integer(
  'image_size', 224, "image size (height, width) in pixels.")
tf.app.flags.DEFINE_integer(
  'train_length', 200000, "number of steps to train for.")
tf.app.flags.DEFINE_integer(
  'save_length', 50000, "number of steps to save for.")
tf.app.flags.DEFINE_integer(
  'batch_size', 128, "batch size.")
tf.app.flags.DEFINE_integer(
  'num_classes', 1000, "number of classes.")
tf.app.flags.DEFINE_integer(
  'print_span', 100, "how long to print.")
tf.app.flags.DEFINE_float(
  'init_learning_rate', 0.0002, 'initial learning rate.')
tf.app.flags.DEFINE_float(
  'moving_average_decay', 0.9999, 'decay to use for the moving average.')
tf.app.flags.DEFINE_boolean(
  'is_training', True, 'whether in a training mode.')

##############################################################################################
# Define the model
##############################################################################################
tf.app.flags.DEFINE_integer(
  'word_embedding_size', 300, "word embedding size in input.")
tf.app.flags.DEFINE_integer(
  'text_embedding_size', 1024, "text embedding size in lstm.")
tf.app.flags.DEFINE_integer(
  'joint_embedding_size', 1024, "embedding size in the visual-semantic space.")
tf.app.flags.DEFINE_integer(
  'max_length', None, "maximal sequence length.")
tf.app.flags.DEFINE_float(
  'weight_decay', 0.00004, 'weight decay on the model weights.')
tf.app.flags.DEFINE_float(
  'text_projection_dropout', None, 'add dropout layer before projection.')
tf.app.flags.DEFINE_string(
  'image_model', "resnet_v1_50", 'image feature extractor.')
tf.app.flags.DEFINE_string(
  'text_model', "lstm", 'text feature extractor.')
tf.app.flags.DEFINE_string(
  'image_feature_name', "global_pool", 'name of image feature.')
tf.app.flags.DEFINE_string(
  'word_embedding_dir', None, 'directory to store pretrained work_embeddings.')
tf.app.flags.DEFINE_boolean(
  'remove_rare_words', False, 'whether to remove the rare words.')
  
##############################################################################################
# Define the loss
##############################################################################################
tf.app.flags.DEFINE_float(
  'margin', 0.2, 'margin on triplet ranking loss.')
tf.app.flags.DEFINE_boolean(
  'semi_hard', True, 'whether to use semi-hard negatives.')

##############################################################################################
# Define the directories
##############################################################################################
tf.app.flags.DEFINE_string(
  'checkpoint_dir', None, 'directory to save the checkpoint.')
tf.app.flags.DEFINE_string(
  'checkpoint_dir_stage1', None, 'directory to save the checkpoint.')
tf.app.flags.DEFINE_string(
  'pretrain_checkpoint_dir', None, 'directory where the pretrained model is saved.') 

##############################################################################################
# Define the input data
##############################################################################################
tf.app.flags.DEFINE_string(
  'dataset', "fashion_iq or shoes")
tf.app.flags.DEFINE_string(
  'data_split', "train", 'either "train" or "test".')
tf.app.flags.DEFINE_string(
  'data_path', None, 'path of dataset.')
tf.app.flags.DEFINE_string(
  'subset', None, 'can be "dress" or "shirt" or "toptee".')
tf.app.flags.DEFINE_integer(
  'threads', 16, "image processing threads.")
tf.app.flags.DEFINE_boolean(
  'constant_lr', True, 'whether to use a constant learning rate.')
tf.app.flags.DEFINE_boolean(
  'augmentation', False, 'whether to use data augmentation.')


FLAGS = tf.app.flags.FLAGS

squeeze = lambda x: tf.squeeze(x, [1, 2])
normalization = lambda x: tf.nn.l2_normalize(x, 1, 1e-10)
expand_dims = lambda x: tf.expand_dims(tf.expand_dims(x, 1), 1)
pairwise_distance = lambda f1, f2, dim: tf.reduce_sum(tf.square(tf.subtract(f1, f2)), dim)


def _build_model(source_images, target_images, modify_texts, seqlengths, vocab_size):
  """ define model & loss """
  with tf.variable_scope(tf.get_variable_scope()):
    cnn_features_source = image_model_ml(source_images)

  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    cnn_features_target = image_model_ml(target_images)

  lstm_features = text_model(modify_texts, seqlengths, vocab_size)
  mod_text = projection_layer(lstm_features, FLAGS.text_projection_dropout, scope='text')

  composite_features = []
  for i in range(len(cnn_features_source)):
    net_name = 'attention' + str(i)
    composite_features.append(attention_layer(cnn_features_source[i], mod_text, scope=net_name))

  losses = []
  for i in range(len(composite_features)):
    f_composite = tf.reduce_mean(composite_features[i], [1, 2])
    f_target = tf.reduce_mean(cnn_features_target[i], [1, 2])
    losses.append(bidirectional_matching_loss(f_composite, f_target))

  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n(losses + reg_losses, name='total_loss')
  
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  
  for l in losses + [total_loss]:
    loss_name = 'matching_loss'
    tf.summary.scalar(loss_name + ' (raw)', l)
    tf.summary.scalar(loss_name, loss_averages.average(l))
  
  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)

  return total_loss, losses

def main():

  if FLAGS.dataset == "fashion_iq":
    trainset = fashion_iq.fashion_iq(path=FLAGS.data_path, split=FLAGS.data_split, subset=FLAGS.subset)
  elif FLAGS.dataset == "shoes":
    trainset = shoes.shoes(path=FLAGS.data_path, split=FLAGS.data_split)
  else:
    raise ValueError("dataset must be fashion_iq or shoes")

  ### initialize the relations between source and target
  if FLAGS.dataset == "fashion_iq":
    trainset.generate_queries_(subset=FLAGS.subset)
    all_texts = trainset.get_all_texts(subset=FLAGS.subset)
  else:
    trainset.generate_queries_()
    all_texts = trainset.get_all_texts()
  num_modif = trainset.num_modifiable_imgs
  max_steps = FLAGS.train_length
  
  vocab = vocabulary.SimpleVocab()
  for text in all_texts:
    vocab.add_text_to_vocab(text)
  if FLAGS.remove_rare_words:
    print('Remove rare words')
    vocab.threshold_rare_words() 
  vocab_size = vocab.get_size()
  print("Number of samples = {}. Number of words = {}.".format(num_modif, vocab_size))

  with tf.Graph().as_default():
    dataset = tf.data.Dataset.from_tensor_slices((trainset.source_files, trainset.target_files, trainset.modify_texts))
    dataset = dataset.prefetch(FLAGS.batch_size).shuffle(num_modif).map(train_pair_image_parse_function, num_parallel_calls=FLAGS.threads).apply(batch_and_drop_remainder(FLAGS.batch_size)).repeat()
    data_iterator = dataset.make_one_shot_iterator()
    batch_source_image, batch_target_image, batch_text = data_iterator.get_next()
  
    source_images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3))
    target_images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3))
    modify_texts_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, None))
    seqlengths_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))

    global_step = tf.train.get_or_create_global_step()

    if FLAGS.constant_lr:
      lr = FLAGS.init_learning_rate
    else:
      boundaries = [int(max_steps * 0.5)]
      values = [FLAGS.init_learning_rate, FLAGS.init_learning_rate * 0.1]
      print('boundaries = %s, values = %s ' % (boundaries, values))
      lr = tf.train.piecewise_constant(global_step, boundaries, values)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    
    with tf.variable_scope(tf.get_variable_scope()):
      total_loss, matching_loss = _build_model(source_images_placeholder, target_images_placeholder, modify_texts_placeholder, seqlengths_placeholder, vocab_size)    
      train_vars = tf.trainable_variables()
      barchnorm = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
      
      # not to train the logits layer
      train_vars = [var for var in train_vars if not "ogits" in var.name]
      barchnorm = [var for var in barchnorm if not "ogits" in var.name]

      barchnorm_op = tf.group(*barchnorm)
      updates_op = tf.assign(global_step, global_step + 1)

    if FLAGS.moving_average_decay:
      ema_op = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay).apply(train_vars)

    with tf.control_dependencies([barchnorm_op, updates_op, ema_op]):
      train_op = opt.minimize(loss=total_loss, global_step=tf.train.get_global_step(), var_list=train_vars)

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summary_op = tf.summary.merge(summaries)

    saver = tf.train.Saver(max_to_keep=6)
    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    sess.run(init_op)
    if FLAGS.checkpoint_dir_stage1:
      load_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir_stage1)
      print("Fine tuning from checkpoint: {}".format(load_checkpoint))
      vars_to_load = optimistic_restore_vars(load_checkpoint)
      finetuning_restorer = tf.train.Saver(var_list=vars_to_load)
      finetuning_restorer.restore(sess, load_checkpoint)

    elif FLAGS.pretrain_checkpoint_dir:
      print("Fine tuning from pretrained checkpoint: {}".format(FLAGS.pretrain_checkpoint_dir))
      checkpoint_vars = tf.train.list_variables(FLAGS.pretrain_checkpoint_dir)
      checkpoint_vars = [v[0] for v in checkpoint_vars]
      vars_can_be_load = []
      all_vars = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
      for i in range(len(all_vars)): 
        var_name = all_vars[i].name.replace(":0", "")
        if (var_name in checkpoint_vars) and (not var_name == "global_step") and (not "ogits" in var_name):
          vars_can_be_load.append(all_vars[i])
      pretrain_restorer = tf.train.Saver(var_list=vars_can_be_load)
      pretrain_restorer.restore(sess, FLAGS.pretrain_checkpoint_dir)      

    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir, graph=sess.graph)

    feed_dict = {
      source_images_placeholder: np.zeros((FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3)), 
      target_images_placeholder: np.zeros((FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3)), 
      modify_texts_placeholder: np.zeros((FLAGS.batch_size, 10), dtype=int),
      seqlengths_placeholder: np.zeros((FLAGS.batch_size), dtype=int)
    }

    tf.train.start_queue_runners(sess=sess)
    start_time = time.time()

    while True:

      source_image_array, target_image_array, raw_text, step = sess.run([batch_source_image, batch_target_image, batch_text, global_step], feed_dict=feed_dict)
      text_array, lengths = vocab.encode_text2id_batch(raw_text)

      if FLAGS.max_length is not None:
        lengths = np.minimum(lengths,  FLAGS.max_length)
        max_length = FLAGS.max_length
      else:
        max_length = max(lengths)

      feed_dict = {
        source_images_placeholder: source_image_array, 
        target_images_placeholder: target_image_array, 
        modify_texts_placeholder: text_array[:, 0:max_length],
        seqlengths_placeholder: lengths
      }

      _, loss_value, matching_loss_value, step = sess.run([train_op, total_loss, matching_loss, global_step], feed_dict=feed_dict)

      if step % FLAGS.print_span == 0:
        duration = time.time() - start_time
        start_time = time.time()
        print("step = %d, total_loss = %.4f, matching_loss = %s, time = %.4f" % (step, loss_value, str(matching_loss_value), duration))
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      if step > 0 and (step % FLAGS.save_length == 0 or step == max_steps):
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step >= max_steps:
        break


if __name__ == '__main__':
  main()
