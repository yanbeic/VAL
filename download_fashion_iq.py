import os
import urllib.request
import tensorflow as tf

### This is a script to download the fashion-iq dataset

# python download_image_data.py --split=0
# python download_image_data.py --split=1
# python download_image_data.py --split=2

tf.app.flags.DEFINE_integer('split', 0, "split")

FLAGS = tf.app.flags.FLAGS

readpath = ['datasets/fashion_iq/image_tag_dataset/image_url/asin2url.dress.txt', \
            'datasets/fashion_iq/image_tag_dataset/image_url/asin2url.shirt.txt', \
            'datasets/fashion_iq/image_tag_dataset/image_url/asin2url.toptee.txt']

savepath = ['datasets/fashion_iq/image_data/dress', \
            'datasets/fashion_iq/image_data/shirt', \
            'datasets/fashion_iq/image_data/toptee']

missing_file = ['datasets/fashion_iq/missing_dress.log', \
                'datasets/fashion_iq/missing_shirt.log', \
                'datasets/fashion_iq/missing_toptee.log']

k = FLAGS.split

with open(missing_file[k], 'a') as f:
  missing = 0
  file = open(readpath[k], "r")
  lines = file.readlines()
  print(len(lines))
  for i in range(len(lines)):
    try:
      line = lines[i].replace('\n','').split(' \t ')
      url = line[1]
      imgpath = os.path.join(savepath[k], line[0]+'.jpg')
      urllib.request.urlretrieve(url, imgpath)
    except:
      missing += 1
      f.write(imgpath)
      print(imgpath)
      pass

print("missing %d." % missing)
