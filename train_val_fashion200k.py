from datasets import fashion200k
from datasets import vocabulary
from config import *
from model import *

import os
import time
import numpy as np
import tensorflow as tf
import math

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
  'dataset', "fashion200k")
tf.app.flags.DEFINE_string(
  'data_split', "train", 'either "train" or "test".')
tf.app.flags.DEFINE_string(
  'data_path', None, 'path of dataset.')
tf.app.flags.DEFINE_integer(
  'threads', 16, "image processing threads.")
tf.app.flags.DEFINE_boolean(
  'constant_lr', False, 'whether to use a constant learning rate.')
tf.app.flags.DEFINE_boolean(
  'augmentation', False, 'whether to use data augmentation.')


FLAGS = tf.app.flags.FLAGS

squeeze = lambda x: tf.squeeze(x, [1, 2])
normalization = lambda x: tf.nn.l2_normalize(x, 1, 1e-10)
expand_dims = lambda x: tf.expand_dims(tf.expand_dims(x, 1), 1)
pairwise_distance = lambda f1, f2, dim: tf.reduce_sum(tf.square(tf.subtract(f1, f2)), dim)

def _build_model(source_images, target_images, modify_texts, seqlengths, vocab_size):

  source_images.set_shape([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
  target_images.set_shape([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])

  with tf.variable_scope(tf.get_variable_scope()):
    cnn_features_source = image_model_ml(source_images) # dim: 1024

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
  print("Construct dataset")
  if FLAGS.dataset == "fashion200k":
    trainset = fashion200k.fashion200k(path=FLAGS.data_path, split=FLAGS.data_split)
  else:
    raise ValueError("dataset must be 'fashion200k'.")

  ### initialize the relations between source and target
  trainset.caption_index_init_()
  max_steps = FLAGS.train_length

  vocab = vocabulary.SimpleVocab()
  all_texts = trainset.get_all_texts()

  for text in all_texts:
    vocab.add_text_to_vocab(text)
  if FLAGS.remove_rare_words:
    print('Remove rare words')
    vocab.threshold_rare_words()
  vocab_size = vocab.get_size()

  with tf.Graph().as_default():
    source_files_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='source_files')
    target_files_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='target_files')
    batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
    modify_texts_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, None), name='modify_text')
    seqlengths_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name='seqlengths')

    input_queue = data_flow_ops.FIFOQueue(capacity=10000, dtypes=[tf.string, tf.string], shapes=[(1,), (1,)])
    enqueue_op = input_queue.enqueue_many([source_files_placeholder, target_files_placeholder])

    nrof_preprocess_threads = 4
    images_batch = []
    for _ in range(nrof_preprocess_threads):
      filenames_s, filenames_t = input_queue.dequeue()
      images1 = []
      for filename in tf.unstack(filenames_s):
        images1.append(train_image_parse_function(filename))
      images2 = []
      for filename in tf.unstack(filenames_t):
        images2.append(train_image_parse_function(filename))
      images_batch.append([images1, images2])

    source_images, target_images = tf.train.batch_join(
      images_batch,
      batch_size=batch_size_placeholder,
      shapes=[(FLAGS.image_size, FLAGS.image_size, 3), (FLAGS.image_size, FLAGS.image_size, 3)],
      capacity=4 * nrof_preprocess_threads * FLAGS.batch_size,
      enqueue_many=True,
      allow_smaller_final_batch=True)
    source_images = tf.identity(source_images, 'source_images')
    target_images = tf.identity(target_images, 'target_images')

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
      total_loss, matching_loss = _build_model(source_images, target_images, modify_texts_placeholder, seqlengths_placeholder, vocab_size)

      train_vars = tf.trainable_variables()
      barchnorm = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batch norm

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

    saver = tf.train.Saver(max_to_keep=10)
    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
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

    """ generate the random source-target pairs per epoch """
    num_iters = 0
    trainset.generate_random_train_queries_(n_modifications_per_image=1)
    num_iter_per_epoch = math.floor(len(trainset.source_files) / FLAGS.batch_size)

    tf.train.start_queue_runners(sess=sess)

    while True:
      start_time = time.time()

      """ Define a sampler: shuffle the random pairs each epoch """
      if num_iters + 1 == num_iter_per_epoch:
        trainset.generate_random_train_queries_(n_modifications_per_image=1)
        num_iter_per_epoch = math.floor(len(trainset.source_files) / FLAGS.batch_size)
        num_iters = 0

      source_files = trainset.source_files[num_iters * FLAGS.batch_size:(num_iters + 1) * FLAGS.batch_size]
      target_files = trainset.target_files[num_iters * FLAGS.batch_size:(num_iters + 1) * FLAGS.batch_size]
      modify_texts = trainset.modify_texts[num_iters * FLAGS.batch_size:(num_iters + 1) * FLAGS.batch_size]
      num_iters = num_iters + 1

      source_files = np.array(source_files)
      source_files = np.expand_dims(source_files, 1)
      target_files = np.array(target_files)
      target_files = np.expand_dims(target_files, 1)
      text_array, lengths = vocab.encode_text2id_batch(modify_texts)

      sess.run(enqueue_op, {source_files_placeholder: source_files, target_files_placeholder: target_files})

      feed_dict = {
        batch_size_placeholder: FLAGS.batch_size,
        modify_texts_placeholder: text_array,
        seqlengths_placeholder: lengths,
      }

      _, loss_value, matching_loss_value, step = sess.run([train_op, total_loss, matching_loss, global_step], feed_dict=feed_dict)

      duration = time.time() - start_time

      if step % FLAGS.print_span == 0:
        print("step = %d, total_loss = %.4f, matching_loss = %s, time = %.4f" % (step, loss_value, str(matching_loss_value), duration))
        ### Need to run enqueue_op again to get data before compute the loss for summary
        sess.run(enqueue_op, {source_files_placeholder: source_files, target_files_placeholder: target_files})
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      if step > 0 and (step % FLAGS.save_length == 0 or step == max_steps):
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step >= max_steps:
        break


if __name__ == '__main__':
  main()
