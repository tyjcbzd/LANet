# 📄 :page_facing_up: LANet: Lightweight Attention Network for Medical Image Segmentation 
This is the official implementation for article "LANet: Lightweight Attention Network for Medical Image Segmentation". 
The article is submitted in Springer proceedings of the ITTA-2024 conference (https://itta.cyber.az).

## Overview
LANet, a Lightweight Attention Network, are presented in the paper and incorporates an Efficient Fusion Attention (EFA) block and an Adaptive Feature Fusion (AFF) decoding block. The EFA block enhances the model's feature extraction capability by capturing task-relevant information while reducing redundancy in channel and spatial locations. The AFF decoding block fuses the purified low-level features from the encoder with the sampled features from the decoder, enhancing the network's understanding and expression of input features. Additionally, the model adopts MobileViT as a lightweight backbone network with a small number of parameters, facilitating easy training and faster predictive inference. The efficiency of LANet was evaluated using four public datasets: kvasir-SEG, CVC-clinicDB, CVC-colonDB, and the Data Science Bowl 2018. 
![Image 1](imgs/Overview.png)


## 	📝 :pencil: Requirements
* torch == 2.1.1+cu121
* tensorboard == 2.11.2
* numpy == 1.24.1
* python == 3.9.18
* torchvision == 0.16.1+cu121

## 	📊 :bar_chart: Datasets
All datasets used in paper are public, you can download online.

Split the datasets for train, validation and test with ratio 8:1:1

##  📈 :chart_with_upwards_trend: Results

### Compare with other SOTA 
1.Quantitative results


2. Qualitative results

![Image 2](imgs/img_qualitative.png)


### Ablation study
![Image 3](imgs/img_ablation.png)

