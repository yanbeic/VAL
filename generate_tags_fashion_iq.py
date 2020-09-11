import json
import os

#### generate text descriptions based on the attribute tags of each asin
readpaths = [
  ['datasets/fashion_iq/image_tag_dataset/tags/asin2attr.dress.train.json', \
  'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.shirt.train.json', \
  'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.toptee.train.json'],
  ['datasets/fashion_iq/image_tag_dataset/tags/asin2attr.dress.val.json', \
  'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.shirt.val.json', \
  'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.toptee.val.json'],
  ['datasets/fashion_iq/image_tag_dataset/tags/asin2attr.dress.test.json', \
  'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.shirt.test.json', \
  'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.toptee.test.json']
]

writepaths = [
  'datasets/fashion_iq/tags/asin2attr-train.txt', 
  'datasets/fashion_iq/tags/asin2attr-val.txt',
  'datasets/fashion_iq/tags/asin2attr-test.txt'
]

####### read raw data, check if exists in folder, and finally write to txt files
folder = 'fashion_iq'
imgfolder = ['dress', 'shirt', 'toptee']

def remove_repplicate_words(text):
  text = text.lower()
  text = text.split()
  return " ".join(sorted(set(text), key=text.index))

def shorten_text(text):
  ### take maximal 20 words
  words = text.split()
  if len(words) > 20:
    # print(len(words))
    words = words[-20::]
    text = " ".join(words)
  return text

for p in range(3):
  readpath = readpaths[p]
  writepath = writepaths[p]
  img_count = 0
  with open(writepath, 'a') as f:
    for k in range(3):
      path = os.path.join('datasets', folder, 'image_data', imgfolder[k])
      imgnames_all = os.listdir(path)
      imgpaths_all = [os.path.join(imgfolder[k], imgname) for imgname in imgnames_all]
      with open(readpath[k]) as handle: 
        dictdump = json.loads(handle.read())
      imagenames = list(dictdump.keys())
      for i in range(len(imagenames)):
        imagename = imagenames[i]
        text = dictdump[imagename]
        text.reverse()
        text = sum(dictdump[imagename], [])
        text = ' '.join(text)
        imagename = imagename + '.jpg'
        if imagename in imgnames_all:
          idx = imgnames_all.index(imagename)
          img_count += 1
          text = text.strip() # clean up strings
          if text != '':
            text = remove_repplicate_words(text)
            text = shorten_text(text)
            f.write('%s;%s \n' % (imgpaths_all[idx], text))
  print(img_count)
