#!/usr/bin/sh
#!/usr/bin/env python
# run fashion200k on VAL without GloVe feature initialization

export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8 # important: used to read the text file in python3 

DATASET=fashion200k
PRETRAIN_DIR='pretrain_model/mobilenet_v1/mobilenet_v1_1.0_224.ckpt'
CNN='mobilenet_v1_ml'
IMG_SIZE=224
AUGMENT=False
TEXT_MODEL=lstm
MODEL_NAME=val_${CNN}

STAGE1_DIR='save_model/fashion200k/'${MODEL_NAME}
DATA_DIR='datasets/fashion200k'
FEAT_DIR='save_model/fashion200k/'${MODEL_NAME}
TEXT_SIZE=1024
JOINT_SIZE=512
WORD_SIZE=512

python train_val_fashion200k.py \
  --checkpoint_dir=${STAGE1_DIR} \
  --pretrain_checkpoint_dir=${PRETRAIN_DIR} \
  --image_model=${CNN} \
  --data_path=${DATA_DIR} \
  --augmentation=${AUGMENT} \
  --image_size=${IMG_SIZE} \
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
  --train_length=160000 \
  --image_feature_name='before_pool'

for i in {25,26}
do
  python extract_features_val.py \
    --checkpoint_dir=${STAGE1_DIR} \
    --data_path=${DATA_DIR} \
    --feature_dir=${FEAT_DIR} \
    --image_model=${CNN} \
    --image_size=${IMG_SIZE} \
    --text_model=${TEXT_MODEL} \
    --joint_embedding_size=${JOINT_SIZE} \
    --word_embedding_size=${WORD_SIZE} \
    --text_embedding_size=${TEXT_SIZE} \
    --query_images=True \
    --text_projection_dropout=0.9 \
    --dataset=${DATASET} \
    --remove_rare_words=False \
    --image_feature_name='before_pool' \
    --exact_model_checkpoint=model.ckpt-${i}0000 

  python extract_features_val.py \
    --checkpoint_dir=${STAGE1_DIR} \
    --data_path=${DATA_DIR} \
    --feature_dir=${FEAT_DIR} \
    --image_model=${CNN} \
    --image_size=${IMG_SIZE} \
    --text_model=${TEXT_MODEL} \
    --joint_embedding_size=${JOINT_SIZE} \
    --word_embedding_size=${WORD_SIZE} \
    --text_embedding_size=${TEXT_SIZE} \
    --query_images=False \
    --text_projection_dropout=0.9 \
    --dataset=${DATASET} \
    --remove_rare_words=False \
    --image_feature_name='before_pool' \
    --exact_model_checkpoint=model.ckpt-${i}0000 
    
  python test_val.py --feature_dir=${FEAT_DIR} --batch_size=20 --dataset=${DATASET} --subset=${SUBSET} 
done
