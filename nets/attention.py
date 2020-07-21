from nets import nets_factory
import tensorflow as tf


slim = tf.contrib.slim

expand_dims = lambda x: tf.expand_dims(tf.expand_dims(x, 1), 1)
squeeze = lambda x: tf.squeeze(x, [1, 2])

def self_attention(features, images, num_heads):
  batch_size, h, w, img_channels = images.get_shape().as_list()
  location_num = h * w
  hidden_size = img_channels // num_heads

  keys = tf.layers.dense(inputs=features, units=hidden_size, use_bias=False)
  values = tf.layers.dense(inputs=features, units=hidden_size, use_bias=False)
  queries = tf.layers.dense(inputs=features, units=hidden_size, use_bias=False)

  keys = tf.reshape(keys, [batch_size, location_num, hidden_size])
  values = tf.reshape(values, [batch_size, location_num, hidden_size])
  queries = tf.reshape(queries, [batch_size, location_num, hidden_size])

  att_matrix = tf.matmul(keys, values, transpose_b=True) / (hidden_size ** 0.5)
  att_matrix = tf.nn.softmax(att_matrix)
  att_matrix = slim.dropout(att_matrix, keep_prob=0.9, scope='Dropout_1b')

  att_out = tf.matmul(att_matrix, queries)
  att_out = tf.reshape(att_out, [batch_size, h, w, hidden_size])

  return att_out


def attention_model(images, texts, model_name, is_training=False, weight_decay=0.00004, scope="attention"):

  with tf.variable_scope(scope):
    arg_scope = nets_factory.arg_scopes_map[model_name](weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):

        ##############################
        batch_size, h, w, img_channels = images.get_shape().as_list()
        texts = expand_dims(texts)
        texts = tf.tile(texts, multiples=[1, h, w, 1])
        vl_features = tf.concat([images, texts], 3)
        vl_features = slim.conv2d(vl_features, img_channels, [1, 1])

        ##############################
        gate_sqz = tf.reduce_mean(vl_features, [1, 2], keep_dims=True)
        att_ch = slim.conv2d(gate_sqz, img_channels, [1, 1])

        gate_sqz = tf.reduce_mean(vl_features, [3], keep_dims=True)
        filter_size = gate_sqz.get_shape().as_list()[1:3]
        att_sp = slim.conv2d(gate_sqz, 1, filter_size)

        joint_att = tf.sigmoid(att_ch)*tf.sigmoid(att_sp)

        ##############################
        num_heads = 2 # the number of heads is tunable
        vl_features = tf.split(vl_features, num_or_size_splits=num_heads, axis=3)
        self_att = []
        for i in range(len(vl_features)):
          self_att.append(self_attention(vl_features[i], images, num_heads))

        self_att = tf.concat(self_att, axis=3)
        self_att = slim.conv2d(self_att, img_channels, [1, 1])

        ##############################
        joint_w = tf.get_variable('r_weight', [], initializer=tf.constant_initializer(1.0))
        self_w = tf.get_variable('weight', [], initializer=tf.constant_initializer(0.0))

        composite_features = joint_w*joint_att*images + self_w*self_att

  return composite_features


