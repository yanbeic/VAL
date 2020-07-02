import json
import os
import glob
import tensorflow as tf

##############################################################################################
# Define the input data
##############################################################################################
tf.app.flags.DEFINE_string(
  'dataset', "fashion_iq or shoes")

FLAGS = tf.app.flags.FLAGS


if FLAGS.dataset == 'fashion_iq':
  #### generate text descriptions
  readpaths = [
    'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.dress.train.json',
    'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.shirt.train.json',
    'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.toptee.train.json',
    'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.dress.val.json',
    'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.shirt.val.json',
    'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.toptee.val.json',
    'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.dress.test.json',
    'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.shirt.test.json',
    'datasets/fashion_iq/image_tag_dataset/tags/asin2attr.toptee.test.json'
  ]

  writepaths = [
    'datasets/fashion_iq/tags/asin2attr-train-dress.txt',
    'datasets/fashion_iq/tags/asin2attr-train-shirt.txt',
    'datasets/fashion_iq/tags/asin2attr-train-toptee.txt',
    'datasets/fashion_iq/tags/asin2attr-val-dress.txt',
    'datasets/fashion_iq/tags/asin2attr-val-shirt.txt',
    'datasets/fashion_iq/tags/asin2attr-val-toptee.txt',
    'datasets/fashion_iq/tags/asin2attr-test-dress.txt',
    'datasets/fashion_iq/tags/asin2attr-test-shirt.txt',
    'datasets/fashion_iq/tags/asin2attr-test-toptee.txt',
  ]

  ####### read raw data, check if exists in folder, and finally write to txt files
  folder = 'fashion_iq'
  imgfolder = ['dress', 'shirt', 'toptee', 'dress', 'shirt', 'toptee', 'dress', 'shirt', 'toptee']

  def remove_replicate_words(text):
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

  for p in range(len(readpaths)):
    readpath = readpaths[p]
    writepath = writepaths[p]
    img_count = 0
    with open(writepath, 'a') as f:
      path = os.path.join('datasets',folder, 'image_data', imgfolder[p])
      imgnames_all = os.listdir(path)
      imgpaths_all = [os.path.join(imgfolder[p], imgname) for imgname in imgnames_all]
      with open(readpath) as handle:
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
            text = remove_replicate_words(text)
            text = shorten_text(text)
            f.write('%s;%s \n' % (imgpaths_all[idx], text))
    print(img_count)


elif FLAGS.dataset == 'shoes':

  readpath = 'datasets/shoes/captions_shoes.json'

  writepaths = ['datasets/shoes/shoes-tag-dress-train.txt',
                'datasets/shoes/shoes-tag-dress-test.txt']

  img_txt_files = ['datasets/shoes/train_im_names.txt',
                   'datasets/shoes/eval_im_names.txt']

  folder = 'datasets/shoes/attributedata'

  ind = 1

  """Process the txt files """

  text_file = open(img_txt_files[ind], "r")
  imgnames = text_file.readlines()
  imgnames = [imgname.strip('\n') for imgname in imgnames]

  imgfolder = os.listdir(folder)
  imgfolder = [imgfolder[i] for i in range(len(imgfolder)) if 'womens' in imgfolder[i]]

  ### the whole path of each file
  imgimages_all = []
  for i in range(len(imgfolder)):
    path = os.path.join(folder, imgfolder[i])
    imgfiles = [f for f in glob.glob(path + "/*/*.jpg", recursive=True)]
    imgimages_all += imgfiles
  ### without the whole path - only the filename
  imgimages_raw = [os.path.basename(imgname) for imgname in imgimages_all]

  """Process the json files """

  with open(readpath) as handle:
    dictdump = json.loads(handle.read())

  with open(writepaths[ind], 'a') as f:
    for i in range(len(imgnames)):
      ind = [k for k in range(len(dictdump)) if dictdump[k]['ImageName'] == imgnames[i]]
      for k in ind:
        imagename = imgimages_all[imgimages_raw.index(imgnames[i])]
        text = dictdump[k]['Caption'].strip()
        f.write('%s;%s \n' % (imagename, text))


else:
  raise ValueError("dataset is unknown.")
