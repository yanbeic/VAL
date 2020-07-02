from datasets import fashion200k
from datasets import fashion_iq
from datasets import shoes
from datasets import vocabulary

import numpy as np
import tensorflow as tf

# python read_glove.py --dataset='fashion_iq' --data_path='datasets/fashion_iq/image_data'
# python read_glove.py --dataset='shoes' --data_path=''
# python read_glove.py --dataset='fashion200k' --data_path='datasets/fashion200k'

tf.app.flags.DEFINE_string(
  'glove_size', "42B")
tf.app.flags.DEFINE_string(
  'dataset', "fashion200k")
tf.app.flags.DEFINE_string(
  'data_path', None, 'path of dataset.')
tf.app.flags.DEFINE_string(
  'data_split', "train", 'either "train" or "test".')
tf.app.flags.DEFINE_string(
  'subset', None, 'can be "dress" or "shirt" or "toptee".')
tf.app.flags.DEFINE_boolean(
  'remove_rare_words', False, 'whether to remove the rare words.')

FLAGS = tf.app.flags.FLAGS

########### read dataset
print("Construct dataset")
if FLAGS.dataset == "fashion200k":
  trainset = fashion200k.fashion200k(path=FLAGS.data_path, split=FLAGS.data_split)
elif FLAGS.dataset == "fashion_iq":
  trainset = fashion_iq.fashion_iq(path=FLAGS.data_path, split=FLAGS.data_split, subset=FLAGS.subset)
elif FLAGS.dataset == "shoes":
  trainset = shoes.shoes(path=FLAGS.data_path, split=FLAGS.data_split)
else:
  raise ValueError("dataset is unknown.")
num_images = len(trainset.filenames)

### initialize the relations between source and target
if FLAGS.dataset == "fashion_iq":
  trainset.generate_queries_(subset=FLAGS.subset)
  all_texts = trainset.get_all_texts(subset=FLAGS.subset)
elif FLAGS.dataset == "shoes":
  trainset.generate_queries_()
  all_texts = trainset.get_all_texts()
elif FLAGS.dataset == "fashion200k":
  ### initialize the relations between source and target
  trainset.caption_index_init_()
  all_texts = trainset.get_all_texts()
else:
  raise ValueError("dataset is unknown.")

num_modif = trainset.num_modifiable_imgs

vocab = vocabulary.SimpleVocab()

for text in all_texts:
  vocab.add_text_to_vocab(text)
if FLAGS.remove_rare_words:
  print('Remove rare words')
  vocab.threshold_rare_words()
vocab_size = vocab.get_size()
print("Number of samples = {}. Number of words = {}.".format(num_modif, vocab_size))

########### read glove
filename = "glove/glove." + FLAGS.glove_size + ".300d.txt"
glove_vocab = []
glove_embed = []
embedding_dict = {}

file = open(filename, 'r', encoding='UTF-8')

for line in file.readlines():
  row = line.strip().split(' ')
  vocab_word = row[0]
  glove_vocab.append(vocab_word)
  embed_vector = [float(i) for i in row[1:]]  # convert to list of float
  embedding_dict[vocab_word] = embed_vector
  glove_embed.append(embed_vector)


glove_vectors = np.zeros((len(vocab.word2id), len(embedding_dict['the'])))
print(glove_vectors.shape)
all_words = list(vocab.word2id.keys())
glove_all_words = list(embedding_dict.keys())
dim = len(embedding_dict['the'])
mu, sigma = 0, 0.09 # mean and standard deviation

count = 0
for i in range(len(vocab.word2id)):
  word = all_words[i]
  idx = vocab.word2id[word]

  if word not in glove_all_words:
    count = count + 1
    vec = np.random.normal(mu, sigma, dim)
  else:
    vec = np.asarray(embedding_dict[word])
  glove_vectors[idx,:] = vec

filename = 'glove/' + FLAGS.dataset + '.'  + FLAGS.glove_size + '.300d.npy'
print(count)
print(filename)
np.save(filename, glove_vectors)
print('Loaded GLOVE')
file.close()
