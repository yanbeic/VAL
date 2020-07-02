from nets import nets_factory
from nets import rnn
from nets import projection
from nets import attention

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
INF = 1e8


def image_model(images):
  network_fn = nets_factory.get_network_fn(
    FLAGS.image_model,
    num_classes=FLAGS.num_classes,
    weight_decay=FLAGS.weight_decay,
    is_training=FLAGS.is_training)
  _, end_points = network_fn(images)
  if FLAGS.image_feature_name is 'global_pool' or 'before_pool':
    features = end_points[FLAGS.image_feature_name]
  else:
    features = []
    features.append(end_points[FLAGS.image_feature_name])
    features.append(end_points['global_pool'])
  return features


def image_model_ml(images):
  network_fn = nets_factory.get_network_fn(
    FLAGS.image_model,
    num_classes=FLAGS.num_classes,
    weight_decay=FLAGS.weight_decay,
    is_training=FLAGS.is_training)
  _, end_points = network_fn(images)
  features = []
  features.append(end_points['low_level'])
  features.append(end_points['mid_level'])
  features.append(end_points['global_pool'])
  return features


def text_model(texts, seqlengths, vocab_size, scope="RNN"):
  features = rnn.extract_text_features(
    model_name=FLAGS.text_model,
    inputs=texts, 
    sequence_length=seqlengths,
    vocab_size=vocab_size, 
    word_embedding_size=FLAGS.word_embedding_size,
    text_embedding_size=FLAGS.text_embedding_size,
    batch_size=FLAGS.batch_size,
    is_training=FLAGS.is_training,
    word_embedding_dir=FLAGS.word_embedding_dir,
    scope=scope)
  return features


def projection_layer(inputs, dropout_keep_prob=None, scope=None, embedding_size=None):
  if embedding_size is not None:
    embeddings = projection.embedding_layers(
      inputs=inputs,
      model_name=FLAGS.image_model,
      embedding_size=embedding_size,
      dropout_keep_prob=dropout_keep_prob,
      is_training=FLAGS.is_training,
      weight_decay=FLAGS.weight_decay,
      scope=scope)
  else:
    embeddings = projection.embedding_layers(
      inputs=inputs,
      model_name=FLAGS.image_model,
      embedding_size=FLAGS.joint_embedding_size,
      dropout_keep_prob=dropout_keep_prob,
      is_training=FLAGS.is_training,
      weight_decay=FLAGS.weight_decay,
      scope=scope)
  return embeddings


def attention_layer(images, texts, scope="attention"):
  features = attention.attention_model(
    images=images,
    texts=texts,
    model_name=FLAGS.image_model,
    is_training=FLAGS.is_training,
    scope=scope)
  return features

    
#### one line function for computing losses
normalization = lambda x: tf.nn.l2_normalize(x, 1, 1e-10)
hinge_loss = lambda dist1, dist2: tf.reduce_mean(tf.maximum(dist1 - dist2 + FLAGS.margin, 0))
hinge_loss_vse = lambda dist1, dist2: tf.reduce_mean(tf.maximum(dist1 - dist2 + FLAGS.margin_vse, 0))
hinge_loss_keep_dim = lambda dist1, dist2: tf.maximum(dist1 - dist2 + FLAGS.margin, 0)

### two type of distance metrics: 
pairwise_distance = lambda f1, f2, dim: tf.reduce_sum(tf.square(tf.subtract(f1, f2)), dim)
dot_product = lambda f1, f2, dim: tf.reduce_sum(f1 * f2, dim)

### remove considerations of itself as the hardest negatives by setting its distance as INF (very large value)
mask_dist = lambda mask, dist_neg_all, infinity: tf.where(mask, tf.ones_like(dist_neg_all)*infinity, dist_neg_all)


def mask_semi_hard(dist_neg_all, dist_pos):
  print('semi_hard negatives - not to choose the hardest negatives')
  hard_mask = tf.less(dist_neg_all, dist_pos)
  dist_neg_all = mask_dist(hard_mask, dist_neg_all, INF)  
  return dist_neg_all


def bidirectional_matching_loss(images, texts, semi_hard=True, name=None, keep_dim=False):

  images = normalization(images)
  texts = normalization(texts)

  ### Note: hardest negatives do not include itself
  if images.get_shape().as_list()[0] is not None:
    batch_size = images.get_shape().as_list()[0]
    self_mask = tf.cast(tf.eye(batch_size), tf.bool)
  elif texts.get_shape().as_list()[0] is not None:
    batch_size = texts.get_shape().as_list()[0]
    self_mask = tf.cast(tf.eye(batch_size), tf.bool)
  else:
    self_mask = None

  ### higher similarity -> smaller distance score
  """ (1) match image to text """
  dist_pos_img = pairwise_distance(images, texts, 1)
  dist_neg_img_all = pairwise_distance(images[:, tf.newaxis], texts, 2)

  if semi_hard:
    hard_mask = tf.less(dist_neg_img_all, dist_pos_img)
    dist_neg_img_all = mask_dist(hard_mask, dist_neg_img_all, INF)
  if self_mask is None:
    self_mask = tf.equal(dist_neg_img_all, dist_pos_img)

  dist_neg_img_all = mask_dist(self_mask, dist_neg_img_all, INF)
  dist_neg_img = tf.reduce_min(dist_neg_img_all, 1)
  if keep_dim:
    img2text = hinge_loss_keep_dim(dist_pos_img, dist_neg_img)
  else:
    img2text = hinge_loss(dist_pos_img, dist_neg_img)

  """ (2) match text to image """
  dist_pos_text = pairwise_distance(texts, images, 1)
  dist_neg_text_all = pairwise_distance(texts[:, tf.newaxis], images, 2)

  if semi_hard:
    hard_mask = tf.less(dist_neg_text_all, dist_pos_text)
    dist_neg_text_all = mask_dist(hard_mask, dist_neg_text_all, INF)

  dist_neg_text_all = mask_dist(self_mask, dist_neg_text_all, INF)
  dist_neg_text = tf.reduce_min(dist_neg_text_all, 1)
  if keep_dim:
    text2img = hinge_loss_keep_dim(dist_pos_text, dist_neg_text)
  else:
    text2img = hinge_loss(dist_pos_text, dist_neg_text)

  ### add two bidirectional terms
  if name is not None:
    matching_loss = tf.add(img2text, text2img, name=name)
  else:
    matching_loss = img2text + text2img

  return matching_loss
