#!/usr/bin/sh
#!/usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################################
###### fashion_iq

## download image data

mkdir datasets/fashion_iq/image_data
mkdir datasets/fashion_iq/image_data/dress
mkdir datasets/fashion_iq/image_data/shirt
mkdir datasets/fashion_iq/image_data/toptee

python download_fashion_iq.py --split=0
python download_fashion_iq.py --split=1
python download_fashion_iq.py --split=2

## generate some needed 'txt' files
python generate_caption_pairs.py --dataset='fashion_iq'
python generate_tags.py --dataset='fashion_iq'

## generate ".py" files used in test time
mkdir groundtruth
python generate_groundtruth.py --dataset='fashion_iq' --data_path='fashion_iq'
python generate_groundtruth.py --dataset='fashion_iq' --data_path='fashion_iq' --subset=dress
python generate_groundtruth.py --dataset='fashion_iq' --data_path='fashion_iq' --subset=shirt
python generate_groundtruth.py --dataset='fashion_iq' --data_path='fashion_iq' --subset=toptee

########################################################################################################################################
###### fashion200k

## please first download image data from the google drive: https://drive.google.com/drive/folders/0B4Eo9mft9jwoamlYWFZBSHFzV3c
## unzip the file under the directory "datasets/fashion200k"

## generate ".py" files used in test time
python generate_groundtruth.py --dataset='fashion200k'

########################################################################################################################################
###### shoes

## please first download image data here: http://tamaraberg.com/attributesDataset/attributedata.tar.gz
## unzip the file under the directory "datasets/shoes"

## generate some needed 'txt' files
python generate_caption_pairs.py --dataset='shoes'
python generate_tags.py --dataset='shoes'

## generate ".py" files used in test time
python generate_groundtruth.py --dataset='shoes' --data_path=''

########################################################################################################################################
###### download glove features from here: http://nlp.stanford.edu/data/glove.42B.300d.zip
## unzip the file under the directory "glove/"

## generate the pre-trained glove features
python read_glove.py --dataset='fashion_iq' --data_path='datasets/fashion_iq/image_data'
python read_glove.py --dataset='shoes' --data_path=''
python read_glove.py --dataset='fashion200k' --data_path='datasets/fashion200k'
