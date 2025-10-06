# Solar Panel Segmentation employing Deep Learning Techniques

### Introduction
This repository is a PyTorch implementation of a semantic segmentation pipeline applied to solar panel detection. 
- Preliminary classification: a classifier distinguish "positive" images (containing solar panels) from "negative" images
- Semantic segmentation: a segmentation model generates segmentation masks of "positive" samples
We use torchvision EfficientNet-v2S for classification, SegFormer with mit-b2 encoder for segmentation. We used these codes to fine-tune and evaluate these models on solar panel datasets.

### Usage
1. Clone repository
   ```shell
   git clone https://github.com/cozzalberto/segment_pvs.git
   ```
3. Train classifier
4. Train segmentation model
5. Test
