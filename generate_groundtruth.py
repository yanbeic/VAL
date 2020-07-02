from datasets import fashion200k
from datasets import fashion_iq
from datasets import shoes
from config import *
from model import *

import tensorflow as tf
import numpy as np

# python generate_groundtruth.py --dataset='fashion200k'
# python generate_groundtruth.py --dataset='shoes' --data_path=''
# python generate_groundtruth.py --dataset='fashion_iq' --data_path='fashion_iq'
# python generate_groundtruth.py --dataset='fashion_iq' --data_path='fashion_iq' --subset=dress
# python generate_groundtruth.py --dataset='fashion_iq' --data_path='fashion_iq' --subset=shirt
# python generate_groundtruth.py --dataset='fashion_iq' --data_path='fashion_iq' --subset=toptee

tf.app.flags.DEFINE_string(
  'data_path', "datasets/fashion200k", 'path of dataset.')
tf.app.flags.DEFINE_string(
  'data_split', "test")
tf.app.flags.DEFINE_string(
  'dataset', "fashion200k")
tf.app.flags.DEFINE_string(
  'subset', None, 'can be "dress" or "shirt" or "toptee".')
  
def main():

  ### prepare test set
  if FLAGS.dataset == "fashion200k":
    testset = fashion200k.fashion200k(path=FLAGS.data_path, split=FLAGS.data_split)
    filename = "groundtruth/fashion200k_modif_pairs.npy"
  elif FLAGS.dataset == "fashion_iq":
    testset = fashion_iq.fashion_iq(path=FLAGS.data_path, split=FLAGS.data_split, subset=FLAGS.subset)
    if FLAGS.subset is None:  
      filename = "groundtruth/fashion_iq_modif_pairs.npy"
    else:
      filename = "groundtruth/fashion_iq_modif_pairs_" + FLAGS.subset + ".npy"
  elif FLAGS.dataset == "shoes":
    testset = shoes.shoes(path=FLAGS.data_path, split=FLAGS.data_split)
    filename = "groundtruth/shoes_modif_pairs.npy"
  else:
    raise ValueError("dataset is unknown.")

  ### generate source-query pairs at test time
  if FLAGS.dataset == "fashion200k":
    testset.generate_test_queries_()
    num_query = len(testset.test_queries)
    num_images = len(testset.filenames)
    groundtruth = np.full((num_query, num_images), False, dtype=bool)

    ### find the matching text pairs in the testset
    for i in range(num_query):
      ### the groundtruth has the same target text :)
      indices = [index for (index, letter) in enumerate(testset.texts) if letter == testset.test_queries[i]['target_caption']]
      groundtruth[i, indices] = True #1
    np.save(filename, groundtruth)

  elif FLAGS.dataset == 'shoes':
    testset.generate_queries_()
    testset.generate_test_images_all_()
    database = testset.database
    num_images = len(database)
    num_query = len(testset.source_files)
    groundtruth = np.full((num_query, num_images), False, dtype=bool)

    for i in range(num_query):
      idx = database.index(testset.target_files[i])
      groundtruth[i, idx] = True
    print('num_images = %d; num_query = %d' % (num_images, num_query))
    np.save(filename, groundtruth)

  elif FLAGS.dataset == 'fashion_iq':
    testset.generate_queries_(subset=FLAGS.subset)
    testset.generate_test_images_all_(subset=FLAGS.subset)
    database = testset.database
    num_images = len(database)
    num_query = len(testset.source_files)
    groundtruth = np.full((num_query, num_images), False, dtype=bool)

    for i in range(num_query):
      idx = database.index(testset.target_files[i])
      groundtruth[i, idx] = True
    print('num_images = %d; num_query = %d' % (num_images, num_query))
    np.save(filename, groundtruth)

if __name__ == '__main__':
  main()
