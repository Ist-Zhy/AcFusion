# AcFusion:Infrared and visible image fusion based on self-attention and convolution with enhanced information extraction
This is the official PyTorch implementation of AcFusion:Infrared and visible image fusion based on self-attention and convolution with enhanced information extraction. We use the following datasets in this post:
* [MSRS](https://github.com/Linfeng-Tang/PIAFusion)
* [RoadScene](https://github.com/hanna-xu/RoadScene)
* [TNO](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029)
* [M3FD](https://github.com/dlut-dimt/TarDAL)
  
Links to the checkpoints can be found in the inference description below.

## Abstract
The purpose of fusing infrared and visible images is to create a single fused image that not only includes texture details and prominent objects but also being well-suited for further intelligent processing. Existing fusion methods often use local convolution, which fails to account for multi-scale and global feature dependencies, while Transformer-based approaches typically face constraints related to computational resources and input image size. To this end, we propose an innovative and streamlined fusion network, called AcFusion. Firstly, we introduce ACmix, which consists of a combination of convolution and multi-head self-attention, to enhance the global modeling capability while retaining as much meaningful information as possible from the source image for further processing. We design the Sobel operator-based attention gradient block (SWGD) to suppress loss of high-frequency information and enhance fine-grained information of the mode. Furthermore, we design a Residual-Dense Attention Block (RDAB) using SWGD as a component, which further improves the representation ability of features through residual connection. We conduct generalization experiments and ablation experiments to demonstrate the effectiveness of our fusion method in integrating information from different modalities. Moreover, we also verify the superiority of our method in the downstream task of object detection.


## Architecture
![](https://github.com/Ist-Zhy/AcFusion/blob/main/docs/AcFusion.png)
Our AcFusion is implemented in `FusionNetwork.py`.

## Recommended Environment
  *   `torch=1.12.0`
  *   `scipy=1.7.3`
  *   `numpy=1.26.3`
  *   `opencv=4.5.5`

## To Test
python test.py
## To Train
python train.py
## Note this
Our code is based on SeAFusion(https://github.com/Linfeng-Tang/SeAFusion).

