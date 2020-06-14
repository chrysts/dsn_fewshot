
## Adaptive Subspaces for Few-Shot Learning
<img src="https://raw.githubusercontent.com/chrysts/chrysts.github.io/master/images/psn.jpg" width="700" height="200" />
(Left) Matching Network, (Middle) Prototypical Network, (Right) Adaptive Subspace Network/Ours

## OVERVIEW

### Requirements:
- PyTorch 1.0 or above
- Python 3.6

There are two backbones separated in different folders. 
- Conv-4, there are two datasets using this backbone: mini-ImageNet and OpenMIC. 
- ResNet-12, there are three datasets using this backbone: mini-ImageNet, tiered-ImageNet, and Cifar-FS. [COMINGSOON]


## DATASET
- mini-ImageNet: [Google Drive](https://drive.google.com/file/d/1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk) (3 GB)
- tiered-ImageNet: [Google Drive](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07) (12 GB)
- OpenMIC: [Fill the form](http://users.cecs.anu.edu.au/~koniusz/openmic-dataset#openmic_req) and [Download](http://users.cecs.anu.edu.au/~koniusz/openmic-dataset/data/openmic_dsn_fewshot.zip) (0.3 GB)
- CIFAR100: [CS TORONTO](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) (0.2 GB)

## USAGE

Train mini-ImageNet:

``` python3 train_subspace_discriminative.py --data-path 'yourdatafolder' ```

Evaluate mini-ImageNet:

``` python3 test_subspace_discriminative.py --data-path 'yourdatafolder' ```


Train OpenMIC:

```python3 train_subspace_museum.py --data-path 'yourdatafolder'  ```




Cite:

```
@inproceedings{simon2020dsn,
        author       = {C. Simon}, {P. Koniusz}, {R. Nock}, and {M. Harandi}
        title        = {Adaptive Subspaces for Few-Shot Learning},
        booktitle    = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
        year         = 2020
        }
```      





Thank you for the codebases:

https://github.com/jakesnell/prototypical-networks

https://github.com/kjunelee/MetaOptNet




 
