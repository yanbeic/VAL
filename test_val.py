from datasets import fashion200k
from datasets import fashion_iq
from datasets import shoes
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
  'data_path', "datasets/fashion200k", 'path of dataset.')
tf.app.flags.DEFINE_string(
  'data_split', "test", 'either "train" or "test".')
tf.app.flags.DEFINE_boolean(
  'compute_distance', True, 'whether to compute image features on all text images or the query images.')
tf.app.flags.DEFINE_integer(
  'batch_size', 1, "batch size.")
tf.app.flags.DEFINE_boolean(
  'gpu_mode', True, 'whether to compute distances on the gpu or cpu.')
tf.app.flags.DEFINE_boolean(
  'test_texts', False, 'whether to compute text features on all text images or the query images.')
tf.app.flags.DEFINE_boolean(
  'test_joint', False, 'whether to use both image and text features for evaluation.')
tf.app.flags.DEFINE_string(
  'subset', None, "which subset to use (1,2,3) for (dress, shirt, toptee).")

INF=1e8


def main():

  start_time = time.time()

  """ compute distance """
  if FLAGS.compute_distance:
    filename = os.path.join(FLAGS.feature_dir, 'query_images.npy')
    query_images = np.load(filename)
    if FLAGS.test_texts:
      filename = os.path.join(FLAGS.feature_dir, 'test_texts.npy')
    else:
      filename = os.path.join(FLAGS.feature_dir, 'test_images.npy')
    test_images = np.load(filename)

    num_query = len(query_images)
    num_test = len(test_images)
    mod2img_dist = np.zeros(shape=[num_query, num_test], dtype=np.float32)

    if FLAGS.gpu_mode:
      ### compute distances on gpu
      graph = tf.Graph()
      with graph.as_default(), tf.device('/gpu:0'):
        query_images = tf.constant(query_images)
        test_images = tf.constant(test_images)
        
        idx_placeholder = tf.placeholder(tf.int32, shape=(None))
        query_image = tf.gather(query_images, idx_placeholder)
        diff = (query_image[:, tf.newaxis] - test_images) ** 2
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
        mod2img_dist[idx_min:idx_max, :] = sess.run(dist, feed_dict=feed_dict)
        if i == num_iters:
          if last_batch_size > 0:
            idx_min = i*FLAGS.batch_size
            idx_max = num_query 
            feed_dict = {idx_placeholder: np.arange(idx_min, idx_max)}
            mod2img_dist[idx_min:idx_max, :] = sess.run(dist, feed_dict=feed_dict)
          break
    else:
      ### compute distances on cpu
      for i in range(num_query):
        diff = query_images[i, :][np.newaxis, :] - test_images
        dist = np.sum(diff ** 2, 1)
        mod2img_dist[i, :] = dist
    
    ### save pre-computed distance
    if FLAGS.test_texts:
      filename = os.path.join(FLAGS.feature_dir, 'mod2text_dist.npy')
    else:
      if FLAGS.subset is not None:
        ### store subset into invidual files
        filename = os.path.join(FLAGS.feature_dir, 'mod2img_dist_'+ FLAGS.subset + '.npy')
      else:
        filename = os.path.join(FLAGS.feature_dir, 'mod2img_dist.npy')
    np.save(filename, mod2img_dist) 
    
    duration = time.time() - start_time
    print("elapsed time = %.2f second " % duration)
    dist = mod2img_dist

  else:
    if FLAGS.test_joint:
      filename = os.path.join(FLAGS.feature_dir, 'mod2text_dist.npy')
      dist1 = np.load(filename)
      if FLAGS.subset is not None:
        ### store subset into invidual files
        filename = os.path.join(FLAGS.feature_dir, 'mod2img_dist_' + FLAGS.subset + '.npy')
      else:
        filename = os.path.join(FLAGS.feature_dir, 'mod2img_dist.npy')
      dist2 = np.load(filename)
      dist = dist1 + dist2
    else:
      if FLAGS.test_texts:
        filename = os.path.join(FLAGS.feature_dir, 'mod2text_dist.npy')
      else:
        if FLAGS.subset is not None:
          ### store subset into invidual files
          filename = os.path.join(FLAGS.feature_dir, 'mod2img_dist_' + FLAGS.subset + '.npy')
        else:
          filename = os.path.join(FLAGS.feature_dir, 'mod2img_dist.npy')
      dist = np.load(filename)


  """ load groundtruth """
  if FLAGS.dataset == "fashion200k":
    filename = 'groundtruth/fashion200k_modif_pairs.npy'
    testset = fashion200k.fashion200k(path=FLAGS.data_path, split=FLAGS.data_split)
  elif FLAGS.dataset == "fashion_iq":
    if FLAGS.subset is None:  
      filename = "groundtruth/fashion_iq_modif_pairs.npy"
    else:
      filename = "groundtruth/fashion_iq_modif_pairs_" + FLAGS.subset + ".npy"
    testset = fashion_iq.fashion_iq(path=FLAGS.data_path, split=FLAGS.data_split, subset=FLAGS.subset)
  elif FLAGS.dataset == 'shoes':
    filename = 'groundtruth/shoes_modif_pairs.npy'
    testset = shoes.shoes(path=FLAGS.data_path, split=FLAGS.data_split)
  else:
    raise ValueError("dataset is unknown.")
  groundtruth = np.load(filename)


  """ perform retrieval """
  start_time = time.time()
  ### generate source-query pairs at test time
  if FLAGS.dataset == "shoes":
    testset.generate_queries_()
    testset.generate_test_images_all_()
  elif FLAGS.dataset == "fashion200k":
    testset.generate_test_queries_()
  else:
    testset.generate_queries_(subset=FLAGS.subset)
    testset.generate_test_images_all_(subset=FLAGS.subset)

  gt_mask = groundtruth.astype(bool)
  order = np.arange(dist.shape[1])
  recall = np.ones(dist.shape)

  for i in range(len(dist)):
    ### Note: here the searching ones do not include itself
    if FLAGS.dataset == "fashion_iq" or FLAGS.dataset == "shoes":
      idx = testset.database.index(testset.source_files[i])
    else:
      idx = testset.test_queries[i]['source_img_id']
    dist[i, idx] = INF

    rank = np.argsort(dist[i, :])
    gt_label = order[gt_mask[i, :]]
    indexes = []
    for j in range(len(gt_label)):
      indexes.append(np.where(rank==gt_label[j])[0][0])
    recall_atk = min(indexes)
    recall[i,0:recall_atk] = 0

  recall_avg = np.sum(recall, 0)/len(dist)*100
  print("recall: R@1 = %.2f, R@10 = %.2f, R@50 = %.2f" % (recall_avg[0], recall_avg[9], recall_avg[49]))

  r1_num = len(dist)*recall_avg[0]/100
  print("%d out of %d is correct rank1" % (r1_num, len(dist)))

  folder = "results/"

  if FLAGS.dataset == "shoes":
    filename = folder + 'results_shoes.log'
  elif FLAGS.dataset == "fashion200k":
    filename = folder + 'results_fashion200k.log'
  else:
    filename = folder + 'results_fashion_iq.log'

  with open(filename, 'a') as f:
    f.write(FLAGS.feature_dir + "\n")
    if FLAGS.test_texts:
      f.write('text retrieval: ')
    elif FLAGS.test_joint:
      f.write('joint retrieval: ')
    else:
      f.write('image retrieval: ')
    if FLAGS.subset is not None:
      f.write(FLAGS.subset + ' ')
    f.write("recall: R@1 = %.2f, R@10 = %.2f, R@50 = %.2f \n" % (recall_avg[0], recall_avg[9], recall_avg[49]))

  duration = time.time() - start_time
  print("elapsed time = %.2f second " % duration)


if __name__ == '__main__':
  main()
