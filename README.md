# SRGAN
Implementation of SRGAN and SRResNet as defined in [Ledig et. al](https://arxiv.org/abs/1609.04802)
* Note: Still in development

## SRResNet Results
| Dataset  | PSNR   | SSIM |
|:--------:|:------:|:----:|
| Set5     | 30.598 | .889 |
| Set14    | 27.228 | .780 |
| BSD100   | 26.191 | .735 |
| Urban100 | 24.683 | .780 |

## Requirements
* Pytorch
* Torchvision
* Recommended: CUDA 8
* Optional: Datasets (small ones included, e.g., BSD100 and Set14, ImageNet not provided)
