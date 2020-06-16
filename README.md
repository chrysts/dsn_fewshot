
## Adaptive Subspaces for Few-Shot Learning

The repository contains the code for:
<br/>
[Adaptive Subspace for Few-Shot Learning](http://openaccess.thecvf.com/content_CVPR_2020/papers/Simon_Adaptive_Subspaces_for_Few-Shot_Learning_CVPR_2020_paper.pdf)
<br/>
CVPR 2020

<img src="https://raw.githubusercontent.com/chrysts/chrysts.github.io/master/images/psn.jpg" width="700" height="200" />
(Left) Matching Network, (Middle) Prototypical Network, (Right) Adaptive Subspace Network/Ours

## OVERVIEW

### Requirements:
- PyTorch 1.0 or above
- Python 3.6

There are two backbones separated in different folders. 
- Conv-4, there are two datasets using this backbone: mini-ImageNet and OpenMIC. 
- ResNet-12, there are three datasets using this backbone: mini-ImageNet, tiered-ImageNet, and Cifar-FS. 


## DATASET
- mini-ImageNet: [Google Drive](https://drive.google.com/file/d/1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk) (3 GB)
- tiered-ImageNet: [Google Drive](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07) (12 GB)
- OpenMIC: [Fill the request form](http://users.cecs.anu.edu.au/~koniusz/openmic-dataset#openmic_req) and [Download](http://users.cecs.anu.edu.au/~koniusz/openmic-dataset/data/openmic_dsn_fewshot.zip) (0.3 GB)
- CIFAR100: [CS TORONTO](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) (0.2 GB)

## USAGE


#### Conv-4

Train mini-ImageNet:

```     python3 train_subspace_discriminative.py --data-path 'yourdatafolder' ```

Evaluate mini-ImageNet:

```     python3 test_subspace_discriminative.py --data-path 'yourdatafolder' ```


Train OpenMIC:

```     python3 train_subspace_museum.py --data-path 'yourdatafolder'  ```


#### ResNet-12

Note: Training using ResNet-12 requires 4 GPUs with ~10GB/GPU


Set the image folders:
```
_IMAGENET_DATASET_DIR = './miniimagenet/' (in data/mini_imagenet.py)
_TIERED_IMAGENET_DATASET_DIR = '/tieredimagenet/' (in data/tiered_imagenet.py)
_CIFAR_FS_DATASET_DIR = './cifar/CIFAR-FS/' (in data/CIFAR_FS.py)
```


Train mini-ImageNet:

```
  python3 train.py --gpu 0,1,2,3 --save-path "./experiments/miniImageNet_subspace" --train-shot 15 --\
  --head Subspace --network ResNet --dataset miniImageNet --eps 0.1
```

Evaluate mini-ImageNet:

```
  python3 test.py --gpu 0 --load ./experiments/miniImageNet_subspace/best_model.pth --episode 1000 \
  --way 5 --shot 5 --query 15 --head Subspace --network ResNet --dataset miniImageNet
```

```
options --dataset [miniImageNet, tieredImageNet, CIFAR_FS]
```



## Citation:

```
@inproceedings{simon2020dsn,
        author       = {C. Simon}, {P. Koniusz}, {R. Nock}, and {M. Harandi}
        title        = {Adaptive Subspaces for Few-Shot Learning},
        booktitle    = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
        year         = 2020
        }
```      




## Acknowledgement
Thank you for the codebases:

[Prototypical Network](https://github.com/jakesnell/prototypical-networks)

[MetaOpt](https://github.com/kjunelee/MetaOptNet)




 
