from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def _state_size_with_prefix(state_size, prefix=None):

  result_state_size = tensor_shape.as_shape(state_size).as_list()
  if prefix is not None:
    if not isinstance(prefix, list):
      raise TypeError("prefix of _state_size_with_prefix should be a list.")
    result_state_size = prefix + result_state_size
  return result_state_size


def make_variable_state_initializer(**kwargs):

  def variable_state_initializer(shape, batch_size, dtype, index):
    args = kwargs.copy()
    if args.get('name'):
      args['name'] = args['name'] + '_' + str(index)
    else:
      args['name'] = 'init_state_' + str(index)
    args['shape'] = shape
    args['dtype'] = dtype
    var = tf.get_variable(**args)
    var = tf.expand_dims(var, 0)
    var = tf.tile(var, tf.stack([batch_size] + [1] * len(shape)))
    var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
    return var
  return variable_state_initializer


def get_initial_cell_state(cell, initializer, batch_size, dtype):

  state_size = cell.state_size
  if nest.is_sequence(state_size):
    state_size_flat = nest.flatten(state_size)
    init_state_flat = [
        initializer(_state_size_with_prefix(s), batch_size, dtype, i)
        for i, s in enumerate(state_size_flat)]
    init_state = nest.pack_sequence_as(structure=state_size, flat_sequence=init_state_flat)
  else:
    init_state_size = _state_size_with_prefix(state_size)
    init_state = initializer(init_state_size, batch_size, dtype, None)
  return init_state


def rnn_layers(model_name, inputs, sequence_length, batch_size, hidden_state_dimension, dropout_keep_prob=0.999, is_training=True, reuse=False):

  state_initializer = make_variable_state_initializer()
  with tf.variable_scope("bidirectional_lstm", reuse=reuse):
    cell = {}
    initial_state = {}
    if 'bi' in model_name:
      directions = ["forward", "backward"]
      hidden_state_dimension = int(hidden_state_dimension/2)
    else:
      directions = ["forward"]
    print(directions)
    for direction in directions:
      with tf.variable_scope(direction):
        # LSTM or GRU cell
        if 'lstm' in model_name:
          print('lstm')          
          cell[direction] = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_state_dimension, state_is_tuple=True)
        elif 'gru' in model_name:
          print('gru')   
          cell[direction] = tf.contrib.rnn.GRUCell(num_units=hidden_state_dimension)
        else:
          raise ValueError("cell must be either 'lstm' or 'gru'.")
        initial_state[direction] = get_initial_cell_state(cell[direction], state_initializer, batch_size, tf.float32)

    if 'bi' in model_name:
      # bidirection LSTM
      # sequence_length must be provided for tf.nn.bidirectional_dynamic_rnn due to internal bug
      (outputs_forward, outputs_backward), (final_states_forward, final_states_backward) = \
        tf.nn.bidirectional_dynamic_rnn(cell["forward"],
                                        cell["backward"],
                                        inputs=inputs,
                                        dtype=tf.float32,
                                        sequence_length=sequence_length,
                                        initial_state_fw=initial_state["forward"],
                                        initial_state_bw=initial_state["backward"])
      # batch_size * T * 1024
      output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
    else:
      outputs_forward, final_states_forward = \
      tf.nn.dynamic_rnn(cell["forward"],
                        inputs=inputs,
                        dtype=tf.float32,
                        sequence_length=sequence_length,
                        initial_state=initial_state["forward"]) 
      output = outputs_forward
    states = tf.reduce_max(output, axis=1, name='mean_states')
  return states


def extract_text_features(model_name, inputs, sequence_length, vocab_size, word_embedding_size, text_embedding_size, batch_size, is_training, word_embedding_dir=None, scope="RNN"):

  initializer = tf.contrib.layers.xavier_initializer()
  with tf.variable_scope(scope):
    if word_embedding_dir is not None:
      print('load pre-trained embeddings from ' + word_embedding_dir)
      pretrain_words = np.load(word_embedding_dir)
      print(pretrain_words.shape)
      pretrain_words = pretrain_words.astype(np.float32)
      token_embedding_weights = tf.get_variable(name="token_embedding_weights", 
                                                dtype=tf.float32,
                                                initializer=pretrain_words)
      print(token_embedding_weights)
    else:
      token_embedding_weights = tf.get_variable(name="token_embedding_weights",
                                                shape=[vocab_size, word_embedding_size],
                                                initializer=initializer)                                   
    token_lstm_input = tf.nn.embedding_lookup(token_embedding_weights, inputs)

    batch_size = inputs.get_shape().as_list()[0]
    states = rnn_layers(
      model_name=model_name,
      inputs=token_lstm_input,
      sequence_length=sequence_length, 
      batch_size=batch_size,
      hidden_state_dimension=text_embedding_size,
      is_training=is_training)
    features = tf.expand_dims(tf.expand_dims(states, 1), 1)  # batch_size * 1 x 1 x 1024
  return features
