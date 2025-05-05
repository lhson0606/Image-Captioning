# Image Captioning Project

## Overview

This project implements an image captioning model based on [Show, Attend and Tell github repos](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning?tab=readme-ov-file)

Last updated: 08/04/2025

## Dependencies

Included in `requirements.txt`:

- torch
- torchvision
- opencv-python
- h5py

## Installation

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## User Guide

Run the following command to generate captions for an image:

```bash
python caption.py -i reports/attachments/test00.jpg -m checkpoints/best_model.pth.tar -wm _data\flickr8k\WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json -b 5
```

## References

- [a PyTorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning?tab=readme-ov-file)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- [Image Captioning with Visual Attention](https://arxiv.org/abs/1502.03044)
