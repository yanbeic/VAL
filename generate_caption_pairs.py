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

  readpaths = [
    'datasets/fashion_iq/captions/cap.dress.val.json',
    'datasets/fashion_iq/captions/cap.shirt.val.json',
    'datasets/fashion_iq/captions/cap.toptee.val.json',
    'datasets/fashion_iq/captions/cap.dress.train.json',
    'datasets/fashion_iq/captions/cap.shirt.train.json',
    'datasets/fashion_iq/captions/cap.toptee.train.json'
  ]

  writepaths = [
    'datasets/fashion_iq/captions_pairs/fashion_iq-val-cap-dress.txt',
    'datasets/fashion_iq/captions_pairs/fashion_iq-val-cap-shirt.txt',
    'datasets/fashion_iq/captions_pairs/fashion_iq-val-cap-toptee.txt',
    'datasets/fashion_iq/captions_pairs/fashion_iq-train-cap-dress.txt',
    'datasets/fashion_iq/captions_pairs/fashion_iq-train-cap-shirt.txt',
    'datasets/fashion_iq/captions_pairs/fashion_iq-train-cap-toptee.txt'
  ]

  imgfolder = ['dress', 'shirt', 'toptee', 'dress', 'shirt', 'toptee']
  folder = 'fashion_iq'

  ####### read raw data, check if exists in folder, and finally write to txt files
  def write_to_file(dictdump, f, imgnames_all, imgpaths):
    num_pairs = 0
    for i in range(len(dictdump)):
      source_imagename = dictdump[i]['candidate'] + '.jpg'
      target_imagename = dictdump[i]['target'] + '.jpg'
      if (source_imagename in imgnames_all) and (target_imagename in imgnames_all):
        idx_s = imgnames_all.index(source_imagename)
        idx_t = imgnames_all.index(target_imagename)
        num_pairs += 1
        texts = dictdump[i]['captions']
        for j in range(len(texts)):
          texts[j] = texts[j].strip()
          if texts[j] != '':
            try:
              f.write('%s;%s;%s \n' % (imgpaths[idx_s], imgpaths[idx_t], texts[j]))
            except:
              print('cannot ')
    return num_pairs

  for p in range(len(readpaths)):
    readpath = readpaths[p]
    writepath = writepaths[p]
    num_pairs = 0
    with open(writepath, 'a') as f:
      path = os.path.join('datasets', folder, 'image_data', imgfolder[p])
      imgnames_all = os.listdir(path)
      imgpaths_all = [os.path.join(imgfolder[p], imgname) for imgname in imgnames_all]
      with open(readpath) as handle:
        dictdump = json.loads(handle.read())
      num_pairs = write_to_file(dictdump, f, imgnames_all, imgpaths_all)
      print(readpaths[p])
      print(num_pairs)


elif FLAGS.dataset == 'shoes':

  readpath = 'datasets/shoes/relative_captions_shoes.json'

  writepaths = ['datasets/shoes/shoes-cap-train.txt',
                'datasets/shoes/shoes-cap-test.txt']

  img_txt_files = ['datasets/shoes/train_im_names.txt',
                   'datasets/shoes/eval_im_names.txt']

  folder = 'datasets/shoes/attributedata'

  ind = 0

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
      ind = [k for k in range(len(dictdump))
             if dictdump[k]['ImageName'] == imgnames[i]
             or dictdump[k]['ReferenceImageName'] == imgnames[i]]
      for k in ind:
        # either belong to the target image ('ImageName')
        # or reference image ('ReferenceImageName')
        if imgnames[i] == dictdump[k]['ImageName']:
          target_imagename = imgimages_all[imgimages_raw.index(imgnames[i])]
          source_imagename = imgimages_all[imgimages_raw.index(dictdump[k]['ReferenceImageName'])]
        else:
          source_imagename = imgimages_all[imgimages_raw.index(imgnames[i])]
          target_imagename = imgimages_all[imgimages_raw.index(dictdump[k]['ImageName'])]
        text = dictdump[k]['RelativeCaption'].strip()
        f.write('%s;%s;%s \n' % (source_imagename, target_imagename, text))


else:
  raise ValueError("dataset is unknown.")
