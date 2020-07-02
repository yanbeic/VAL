#!/usr/bin/sh
#!/usr/bin/env python

export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8 # important: used to read the text file in python3

DATASET=fashion_iq
PRETRAIN_DIR='pretrain_model/resnet_v2_50/resnet_v2_50.ckpt'
CNN='resnet_v2_50_ml'
IMG_SIZE=256
AUGMENT=True
PRE_TEXT=glove/fashion_iq.6B.300d.npy
TEXT_MODEL=lstm
MODEL_NAME=val_${CNN}

STAGE1_DIR='save_model/fashion_iq/'${MODEL_NAME}
DATA_DIR='datasets/fashion_iq/image_data'
FEAT_DIR='save_model/fashion_iq/'${MODEL_NAME}
TEXT_SIZE=1024
JOINT_SIZE=512
WORD_SIZE=300

python train_val.py \
  --checkpoint_dir=${STAGE1_DIR} \
  --pretrain_checkpoint_dir=${PRETRAIN_DIR} \
  --image_model=${CNN} \
  --data_path=${DATA_DIR} \
  --batch_size=32 \
  --print_span=20 \
  --save_length=10000 \
  --text_model=${TEXT_MODEL} \
  --joint_embedding_size=${JOINT_SIZE} \
  --word_embedding_size=${WORD_SIZE} \
  --text_embedding_size=${TEXT_SIZE} \
  --margin=0.2 \
  --text_projection_dropout=0.9 \
  --init_learning_rate=0.0002 \
  --dataset=${DATASET} \
  --train_length=50000 \
  --constant_lr=True \
  --image_size=${IMG_SIZE} \
  --image_feature_name='before_pool' \
  --augmentation=${AUGMENT} \
  --word_embedding_dir=${PRE_TEXT}


for i in 5
do
  for SUBSET in {dress,shirt,toptee}
    do
      python extract_features_val.py \
        --checkpoint_dir=${STAGE1_DIR} \
        --data_path=${DATA_DIR} \
        --feature_dir=${FEAT_DIR} \
        --image_model=${CNN} \
        --text_model=${TEXT_MODEL} \
        --joint_embedding_size=${JOINT_SIZE} \
        --word_embedding_size=${WORD_SIZE} \
        --text_embedding_size=${TEXT_SIZE} \
        --query_images=True \
        --text_projection_dropout=0.9 \
        --dataset=${DATASET} \
        --remove_rare_words=True \
        --image_size=${IMG_SIZE} \
        --image_feature_name='before_pool' \
        --subset=${SUBSET} \
        --exact_model_checkpoint=model.ckpt-${i}0000

      python extract_features_val.py \
        --checkpoint_dir=${STAGE1_DIR} \
        --data_path=${DATA_DIR} \
        --feature_dir=${FEAT_DIR} \
        --image_model=${CNN} \
        --text_model=${TEXT_MODEL} \
        --joint_embedding_size=${JOINT_SIZE} \
        --word_embedding_size=${WORD_SIZE} \
        --text_embedding_size=${TEXT_SIZE} \
        --query_images=False \
        --text_projection_dropout=0.9 \
        --dataset=${DATASET} \
        --remove_rare_words=True \
        --image_size=${IMG_SIZE} \
        --image_feature_name='before_pool' \
        --subset=${SUBSET} \
        --exact_model_checkpoint=model.ckpt-${i}0000
        
      python test_val.py --feature_dir=${FEAT_DIR} --batch_size=20 --dataset=${DATASET} --subset=${SUBSET}
  done
done
