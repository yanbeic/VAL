from nets import nets_factory
import tensorflow as tf

slim = tf.contrib.slim

def embedding_layers(inputs, model_name, embedding_size=512, dropout_keep_prob=None, is_training=False, weight_decay=0.00004, scope=None):
  
    with tf.variable_scope("projection"):
        arg_scope = nets_factory.arg_scopes_map[model_name](weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
          with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net = inputs
            if dropout_keep_prob is not None:
              print('add dropout = %.4f to projection layer' % dropout_keep_prob)
              net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout')
            joint_embeddings = slim.conv2d(net, embedding_size, [1, 1], activation_fn=None, scope=scope)
            joint_embeddings = tf.squeeze(joint_embeddings, [1, 2])

    return joint_embeddings
