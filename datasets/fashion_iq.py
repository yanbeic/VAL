import os
import random

class fashion_iq():
  
  def __init__(self, path, split="train", subset=None):
    super()

    self.split = split 
    self.path = path

    if split == "train":
      if subset is None:
        filename = "datasets/fashion_iq/tags/asin2attr-train.txt"
      else:
        filename = "datasets/fashion_iq/tags/asin2attr-train-" + subset + ".txt"
    elif split == "test":
      if subset is None:
        filename = "datasets/fashion_iq/tags/asin2attr-test.txt"
      else:
        filename = "datasets/fashion_iq/tags/asin2attr-test-" + subset + ".txt"
    else:
      raise ValueError("split must be 'train' or 'test'.")
    
    print(filename)
    file = open(filename, "r")
    lines = file.readlines()

    self.imgs = []
    self.filenames = []
    self.texts = []

    for line in lines:
      line = line.strip('\n').split(';')
      img = {
          'file_path': line[0],
          'captions': [self.caption_post_process(s=line[1])],
      }
      self.filenames += [os.path.join(self.path, img['file_path'])]
      self.texts += img['captions']
      self.imgs += [img]
    print('Number of samples = %d' % len(self.imgs))


  def caption_post_process(self, s):
    return s.strip().replace('.', 'dotmark').replace('?', 'questionmark').replace('&', 'andmark').replace('*', 'starmark')


  def get_all_texts(self, subset=None):
    if subset is None:
      filenames = [
        "datasets/fashion_iq/captions_pairs/fashion_iq-train-cap.txt"
      ]
    else:
      subset = '-' + subset
      filenames = [
        "datasets/fashion_iq/captions_pairs/fashion_iq-train-cap" + subset + ".txt",
      ]

    self.all_texts = []
    lengths = []
    for k in range(len(filenames)):
      filename = filenames[k]
      if "asin2attr" in filename:
        cap_idx = 1
      else:
        cap_idx = 2
      file = open(filename, "r")
      lines = file.readlines()
      for line in lines:
        line = line.strip('\n').split(';')
        line = self.caption_post_process(s=line[cap_idx])
        lengths.append(len(line.split()))
        self.all_texts += [line]
    return self.all_texts


  def generate_queries_(self, subset=None):
    self.source_files =[]
    self.target_files = []
    self.modify_texts = []
    self.source_texts = []
    self.target_texts = []
    self.num_modifiable_imgs = 0
    self.num_no_tags_s = 0
    self.num_no_tags_t = 0
    self.max_length = 0

    if self.split == 'train':
      if subset is None:
        filename = "datasets/fashion_iq/captions_pairs/fashion_iq-train-cap.txt"
      else:
        filename = "datasets/fashion_iq/captions_pairs/fashion_iq-train-cap-" + subset + ".txt"
    elif self.split == 'test':
      if subset is None:
        filename = "datasets/fashion_iq/captions_pairs/fashion_iq-val-cap.txt"
      else:
        filename = "datasets/fashion_iq/captions_pairs/fashion_iq-val-cap-" + subset + ".txt"
    else:
      raise ValueError("split must be 'train' or 'test'.")

    file = open(filename, "r")
    lines = file.readlines()
    for line in lines:
      line = line.split(';')
      self.source_files += [os.path.join(self.path, line[0])]
      self.target_files += [os.path.join(self.path, line[1])]
      line[2] = line[2].strip('\n')
      self.modify_texts += [self.caption_post_process(s=line[2])]
      self.max_length = max(self.max_length, len(self.caption_post_process(s=line[2]).split(' ')))

      if self.source_files[self.num_modifiable_imgs] in self.filenames:
        idx_s = self.filenames.index(self.source_files[self.num_modifiable_imgs])
        self.source_texts += [self.texts[idx_s]]
      else:
        self.source_texts += ['']
        self.num_no_tags_s += 1

      if self.target_files[self.num_modifiable_imgs] in self.filenames:
        idx_t = self.filenames.index(self.target_files[self.num_modifiable_imgs])
        self.target_texts += [self.texts[idx_t]]
      else:
        self.target_texts += ['']
        self.num_no_tags_t += 1

      self.num_modifiable_imgs += 1


  def filter_to_get_useful_images_(self):
    useful_files = self.source_files + self.target_files
    self.imgs = [self.imgs[i] for i in range(len(self.filenames)) if self.filenames[i] in useful_files]
    self.texts = [self.texts[i] for i in range(len(self.filenames)) if self.filenames[i] in useful_files]
    self.filenames = [self.filenames[i] for i in range(len(self.filenames)) if self.filenames[i] in useful_files]


  def random_shuffle_pairs_(self):
    shuffle_idx = list(range(len(self.source_files)))
    random.shuffle(shuffle_idx)
    self.source_files = [self.source_files[i] for i in shuffle_idx]
    self.target_files = [self.target_files[i] for i in shuffle_idx]
    self.modify_texts = [self.modify_texts[i] for i in shuffle_idx]
    self.source_texts = [self.source_texts[i] for i in shuffle_idx]
    self.target_texts = [self.target_texts[i] for i in shuffle_idx]
    print('shuffling the random source-target pairs. It gives %d pairs.' % len(self.source_files))


  def generate_test_images_all_(self, subset=None):
    if subset is not None:
      self.subset = subset
      self.modify_texts = [self.modify_texts[i] for i in range(len(self.source_files)) if self.subset in self.source_files[i]]
      self.source_files = [source_file for source_file in self.source_files if self.subset in source_file]
      self.target_files = [target_file for target_file in self.target_files if self.subset in target_file]
    all_files = self.source_files + self.target_files
    ### get unique images
    self.database = list(set(all_files))
    self.database = sorted(self.database)
    print(self.database[0])
