# SRGAN
Implementation of SRGAN and SRResNet as defined in [Ledig et. al](https://arxiv.org/abs/1609.04802)

## SRResNet Results
|          | Ours          | Reported     |
| Dataset  | PSNR   | SSIM | PSNR  | SSIM |
|:--------:|:------:|:----:|:-----:|:----:|
| Set5     | 30.598 | .889 | 32.05 | .902 |
| Set14    | 27.228 | .780 | 28.49 | .818 |
| BSD100   | 26.191 | .735 | 27.58 | .762 |
| Urban100 | 24.683 | .780 | N/A   | N/A  |

## SRGAN Results
|          | Ours          | Reported     |
| Dataset  | PSNR   | SSIM | PSNR  | SSIM |
|:--------:|:------:|:----:|:-----:|:----:|
| Set5     | 28.650 | .800 | 29.40 | .847 |
| Set14    | 25.719 | .681 | 26.02 | .740 |
| BSD100   | 24.722 | .620 | 25.15 | .669 |
| Urban100 | 23.259 | .668 | N/A   | N/A  |

## Requirements
* Pytorch
* Torchvision
* Recommended: CUDA 8
* Optional: Datasets (small ones included, e.g., BSD100 and Set14, ImageNet not provided)
