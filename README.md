# Visiolinguistic-Attention-Learning

[Tensorflow](https://www.tensorflow.org/) code of VAL model

[Chen et al. Image Search with Text Feedback by Visiolinguistic Attention Learning. CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Image_Search_With_Text_Feedback_by_Visiolinguistic_Attention_Learning_CVPR_2020_paper.pdf)


## Getting Started

### Prerequisites:

- Datasets: [Fashion200k](https://github.com/xthan/fashion-200k) [1], [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq) [2], [Shoes](http://tamaraberg.com/attributesDataset/index.html) [3,4].
- Python 3.6.8
- Tensorflow 1.10.0


### Preparation:

(1) Download ImageNet pretrained models: [mobilenet](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) and 
[resnet](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz), which should be put under the directory `pretrain_model`.

(2) Follow steps in `scripts/prepare_data.sh` to prepare datasets. Note: `fashion200k` and `shoes` can be downloaded manually. Relevant `py` files for data preparation are detailed below.
* `download_fashion_iq.py`: crawl the image data from Amazon websites. Note that some url links might be broken. 
* `generate_groundtruth.py`: generate some `.npy` files that charaterize the groundtruth annotations during test time. 
* `read_glove.py`: prepare the pre-trained `glove` word embeddings to initialize the text model (i.e. LSTM). 

## Running Experiments

### Training & Testing: 

Train and test the VAL model on different datasets in one script file as follows.
<!-- On `fashion200k`, run -->
```
bash scripts/run_fashion200k.sh
```
<!-- On `fashion_iq`, run -->
```
bash scripts/run_fashion_iq.sh
```
<!-- On `shoes`, run -->
```
bash scripts/run_shoes.sh
```
The test results will be finally reported in `results/results_fashion_iq.log`.

Our implementation include the following `.py` files. Note that `fashion200k` is formated differently compared to `fashion_iq` or `shoes`, as a triplet of source image, text and target image is not pre-given, but is instead sampled randomly during training. Therefore, there are two implementation to build and run the training graph.
* `train_val.py`: build and run the training graph on dataset `fashion_iq` or `shoes`.
* `train_val_fashion200k.py`: build and run the training graph on dataset `fashion200k`.
* `model.py`: define the model and losses.
* `config.py`: define image preprocessing and other configurations.
* `extract_features_val.py`: extract features from the model.
* `test_val.py`: compute distance, perform retrieval, and report results in the `log` file.


## Bibtex:

```
@inproceedings{chen2020image,
  title={Image Search with Text Feedback by Visiolinguistic Attention Learning},
  author={Chen, Yanbei and Gong, Shaogang and Bazzani, Loris},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3001--3011},
  year={2020}
}
```

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.


## References
[1] Automatic Spatially-aware Fashion Concept Discovery, ICCV2019 <br />
[2] The Fashion IQ Dataset: Retrieving Images by Combining Side Information and Relative Natural Language Feedback, CVPRW2019 <br />
[3] Dialog-based interactive image retrieval, NeuRIPS2018 <br />
[4] Auomatic attribute discovery and characterization from noisy web data, ECCV10 <br />
