
## Adaptive Subspaces for Few-Shot Learning

![img](https://raw.githubusercontent.com/chrysts/chrysts.github.io/master/images/psn.jpg) 

### Requirements:
- PyTorch 1.0 or above
- Python 3.6

There are two backbones separated in different folders. 
- Conv-4, there are two datasets using this backbone: mini-ImageNet and OpenMIC. 
- ResNet-12, there are three datasets using this backbone: mini-ImageNet, tiered-ImageNet, and Cifar-FS. [COMINGSOON]

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




 
