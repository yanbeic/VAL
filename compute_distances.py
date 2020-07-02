from config import *
from model import *

import os
import time
import numpy as np
import math

# export CUDA_VISIBLE_DEVICES=2
# python test_modif.py --feature_dir='save_features/fashion200k/tirg_mobilenet_v1_stage2' --batch_size=20

tf.app.flags.DEFINE_string(
  'feature_dir', None, 'directory to extract the feature.')
tf.app.flags.DEFINE_string(
  'dataset', "fashion200k", 'either "fashion200k" or "flickr30k".')
tf.app.flags.DEFINE_string(
  'data_path', "/home/ubuntu/efs/datasets/fashion-200k", 'path of dataset.')
tf.app.flags.DEFINE_string(
  'data_split', "test", 'either "train" or "test".')
tf.app.flags.DEFINE_string(
  'metrics', 'l2', 'Type of metrics to compute distance: can be "l2" or "dot_product".')
tf.app.flags.DEFINE_boolean(
  'compute_distance', True, 'whether to compute image features on all text images or the query images.')
tf.app.flags.DEFINE_integer(
  'batch_size', 1, "batch size.")
tf.app.flags.DEFINE_boolean(
  'gpu_mode', True, 'whether to compute distances on the gpu or cpu.')

tf.app.flags.DEFINE_string(
  'subset', None, 'can be "dress" or "shirt" or "toptee".')

INF=1e8

def main():
  start_time = time.time()
  """ compute or load pairwise distances """

  if FLAGS.subset is not None:
    subset = FLAGS.subset 
  else:
    subset = ''

  ### load pre-computed features
  filename = os.path.join(FLAGS.feature_dir, subset, 'texts.npy')
  queries = np.load(filename)
  filename = os.path.join(FLAGS.feature_dir, subset, 'images.npy')
  tests = np.load(filename)

  num_query = len(queries)
  num_test = len(tests)
  distances = np.zeros(shape=[num_query, num_test], dtype=np.float32)

  if FLAGS.gpu_mode:
    ### compute distances on gpu
    graph = tf.Graph()
    with graph.as_default(), tf.device('/gpu:0'):
      queries = tf.constant(queries)
      tests = tf.constant(tests)
      
      idx_placeholder = tf.placeholder(tf.int32, shape=(None))
      query = tf.gather(queries, idx_placeholder)
      diff = (query[:, tf.newaxis] - tests) ** 2
      dist = tf.reduce_sum(diff, 2)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth=True

    sess = tf.Session(graph=graph, config=config)

    i = 0
    num_iters = math.floor(num_query/FLAGS.batch_size)  
    last_batch_size = num_query - num_iters*FLAGS.batch_size
    print('num_iters = %d, last_batch_size = %d' % (num_iters, last_batch_size))

    while True:
      idx_min = i*FLAGS.batch_size
      idx_max = (i+1)*FLAGS.batch_size
      i = i + 1
      feed_dict = {idx_placeholder: np.arange(idx_min, idx_max)}
      distances[idx_min:idx_max, :] = sess.run(dist, feed_dict=feed_dict)
      if i == num_iters:
        if last_batch_size > 0:
          idx_min = i*FLAGS.batch_size
          idx_max = num_query 
          feed_dict = {idx_placeholder: np.arange(idx_min, idx_max)}
          distances[idx_min:idx_max, :] = sess.run(dist, feed_dict=feed_dict)
        break
  else:
    ### compute distances on cpu
    # here compute the similarity scores
    for i in range(num_query):
      diff = queries[i, :][np.newaxis, :] - tests
      dist = np.sum(diff ** 2, 1)
      distances[i, :] = dist
  
  ### save pre-computed distance
  filename = os.path.join(FLAGS.feature_dir, 'text2img_dist.npy')
  np.save(filename, distances) 
  
  duration = time.time() - start_time
  print("elapsed time = %.2f second " % duration)
  dist = distances
  

if __name__ == '__main__':
  main()

