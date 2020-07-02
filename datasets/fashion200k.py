import os
import glob
import random
import io

class fashion200k():

  def __init__(self, path, split="train"):
    super()

    self.split = split 
    self.path = path

    label_path = "datasets/fashion200k/labels/"
    print("Processing {} set".format(split))
    label_files = glob.glob(label_path + "*_" + split + "_*.txt")
    label_files.sort()

    def caption_post_process(s):
      return s.strip().replace('.', 'dotmark').replace('?', 'questionmark').replace('&', 'andmark').replace('*', 'starmark')

    self.imgs = []
    self.filenames = []
    self.texts = []

    for label_file in label_files:
      print('read ' + label_file)
      with io.open(label_file, "r", encoding="utf8") as fd:
        for line in fd.readlines():
          line = line.split("\t")
          img = {
              'file_path': line[0],
              'captions': [caption_post_process(line[2])],
              'modifiable': False
          }
          self.filenames += [os.path.join(self.path, img['file_path'])]
          self.texts += img['captions']
          self.imgs += [img]


  def get_all_texts(self, get_modifiable=False):
    texts = []
    if get_modifiable is False:
      for img in self.imgs:
        for c in img['captions']:
          texts.append(c)
    else:
      imgs_mod = [self.imgs[i] for i in range(len(self.imgs)) if self.imgs[i]['modifiable']]
      for img in imgs_mod:
         for c in img['captions']:
           texts.append(c)
    return texts


  def get_different_word(self, source_caption, target_caption):
    source_words = source_caption.split()
    target_words = target_caption.split()
    for source_word in source_words:
      if source_word not in target_words:
        break
    for target_word in target_words:
      if target_word not in source_words:
        break
    mod_str = 'replace ' + source_word + ' with ' + target_word
    return source_word, target_word, mod_str


  def generate_test_queries_(self):
    file2imgid = {}
    for i, img in enumerate(self.imgs):
      file2imgid[img['file_path']] = i
    with open('datasets/fashion200k/test_queries.txt') as f:
      lines = f.readlines()
    self.test_queries = []
    self.query_filenames = []
    self.modify_texts = []
    for line in lines:
      source_file, target_file = line.split()
      idx = file2imgid[source_file]
      target_idx = file2imgid[target_file]
      source_caption = self.imgs[idx]['captions'][0]
      target_caption = self.imgs[target_idx]['captions'][0]
      source_word, target_word, mod_str = self.get_different_word(
          source_caption, target_caption)
      self.test_queries += [{
          'source_img_id': idx,
          'source_caption': source_caption,
          'target_caption': target_caption,
          'mod': {
              'str': mod_str
          }
      }]
      self.query_filenames += [os.path.join(self.path, self.imgs[idx]['file_path'])]
      self.modify_texts += [mod_str]


  def caption_index_init_(self):
    """ index caption to generate training query-target example on the fly"""
    caption2id = {}
    id2caption = {}
    caption2imgids = {}
    for i, img in enumerate(self.imgs):
      for c in img['captions']:
        if not c in caption2id:
          id2caption[len(caption2id)] = c
          caption2id[c] = len(caption2id)
          caption2imgids[c] = []
        caption2imgids[c].append(i)
    self.caption2imgids = caption2imgids
    print('unique cations = %d' % len(caption2imgids))

    parent2children_captions = {}
    for c in caption2id.keys():
      for w in c.split():
        p = c.replace(w, '')
        p = p.replace('  ', ' ').strip()
        if not p in parent2children_captions:
          parent2children_captions[p] = []
        if c not in parent2children_captions[p]:
          parent2children_captions[p].append(c)
    self.parent2children_captions = parent2children_captions

    for img in self.imgs:
      img['modifiable'] = False
      img['parent_captions'] = []
    for p in parent2children_captions:
      if len(parent2children_captions[p]) >= 2:
        for c in parent2children_captions[p]:
          for imgid in caption2imgids[c]:
            self.imgs[imgid]['modifiable'] = True
            self.imgs[imgid]['parent_captions'] += [p]

    num_modifiable_imgs = 0
    for img in self.imgs:
      if img['modifiable']:
        num_modifiable_imgs += 1
    self.num_modifiable_imgs = num_modifiable_imgs
    print('Modifiable images = %d' % num_modifiable_imgs) 


  def caption_index_sample_(self, idx):
    while not self.imgs[idx]['modifiable']:
      idx = np.random.randint(0, len(self.imgs))
    img = self.imgs[idx]
    while True:
      p = random.choice(img['parent_captions'])
      c = random.choice(self.parent2children_captions[p])
      if c not in img['captions']: 
        break
    target_idx = random.choice(self.caption2imgids[c])

    source_caption = self.imgs[idx]['captions'][0]
    target_caption = self.imgs[target_idx]['captions'][0]
    source_word, target_word, mod_str = self.get_different_word(
        source_caption, target_caption)
    return idx, target_idx, source_word, target_word, mod_str


  def filter_to_get_modifiable_images_(self):
    self.filenames = [os.path.join(self.path, self.imgs[i]['file_path']) for i in range(len(self.imgs)) if self.imgs[i]['modifiable']]
    self.texts = [self.imgs[i]['captions'][0] for i in range(len(self.imgs)) if self.imgs[i]['modifiable']]


  def get_unmodifiable_images_(self):
    print('get unmodifiable images')
    self.filenames_um = []
    self.texts_um = []
    for i, img in enumerate(self.imgs):
      if img['modifiable'] is False:
        self.filenames_um += [os.path.join(self.path, img['file_path'])]
        self.texts_um += img['captions']
    print('The amount of unmodifiable images is %d.' % len(self.texts_um))


  def get_item_(self):
    self.items = [x.split('/')[-2] for x in self.filenames]


  def generate_random_train_queries_(self, n_modifications_per_image=3):
    self.source_files =[]
    self.target_files = []
    self.modify_texts = []
    self.source_texts = []
    self.target_texts = []
    already_visited = set()

    for i, img in enumerate(self.imgs):
      if img['modifiable']:
        for j in range(n_modifications_per_image):
          idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(i)
          # ensure the choosen pairs does not share the same words even the ordering is different
          set1 = set(self.imgs[idx]['captions'][0].split(' '))
          set2 = set(self.imgs[target_idx]['captions'][0].split(' '))
          if set1 != set2: 
            key = "{}-{}".format(target_idx, idx)
            inv_key = "{}-{}".format(idx, target_idx)
            if not (key in already_visited or inv_key in already_visited): 
              self.source_files += [os.path.join(self.path, self.imgs[idx]['file_path'])]
              self.target_files += [os.path.join(self.path, self.imgs[target_idx]['file_path'])]
              self.modify_texts += [mod_str]
              self.source_texts += self.imgs[idx]['captions']
              self.target_texts += self.imgs[target_idx]['captions']
              already_visited.add(key)

    # randomly shuffle the epoch wise sampled pairs
    shuffle_idx = list(range(len(self.source_files)))
    random.shuffle(shuffle_idx)
    self.source_files = [self.source_files[i] for i in shuffle_idx]
    self.target_files = [self.target_files[i] for i in shuffle_idx]
    self.modify_texts = [self.modify_texts[i] for i in shuffle_idx]
    self.source_texts = [self.source_texts[i] for i in shuffle_idx]
    self.target_texts = [self.target_texts[i] for i in shuffle_idx]
    print('shuffling the random source-target pairs. It gives %d pairs.' % len(self.source_files))

