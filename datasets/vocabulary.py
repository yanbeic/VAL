import numpy as np
import nltk

class SimpleVocab():

  def __init__(self):
    super()

    self.word2id = {}
    self.wordcount = {}
    self.word2id['<UNK>'] = 0
    self.wordcount['<UNK>'] = 9e9


  def tokenize_text(self, text):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    return tokens


  def add_text_to_vocab(self, text):
    tokens = self.tokenize_text(text)
    for token in tokens:
      if token not in self.word2id:
        self.word2id[token] = len(self.word2id)
        self.wordcount[token] = 0
      self.wordcount[token] += 1


  def threshold_rare_words(self, wordcount_threshold=2):
    for w in self.word2id:
      if self.wordcount[w] < wordcount_threshold:
        self.word2id[w] = 0


  def encode_text(self, text):
    tokens = self.tokenize_text(text)
    x = [self.word2id.get(t, 0) for t in tokens]
    return x


  def encode_text2id_batch(self, batch_text):
    # tokenize text to id
    if type(batch_text[0]) is bytes:
      text_id = [self.encode_text(str(text, 'utf-8')) for text in batch_text]
    elif type(batch_text[0]) is str:
      text_id = [self.encode_text(text) for text in batch_text]
    else:
      raise TypeError('Type of batch_text should be bytes or str.')
    # length of each sentence/phrase
    lengths = [len(t) for t in text_id]
    text_array = np.zeros((len(lengths), np.max(lengths)), dtype=int)
    for i in range(len(text_id)):
      text_array[i,:lengths[i]] = np.array(text_id[i])
    return text_array, np.array(lengths)


  def get_size(self):
    return len(self.word2id)
