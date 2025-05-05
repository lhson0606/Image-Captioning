# Image Captioning Project

## Overview

This project implements an image captioning model based on [Show, Attend and Tell github repos](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning?tab=readme-ov-file)

Last updated: 05/05/2025

## Dependencies

Included in `requirements.txt` or `cv-g10-env.yml`:

- torch==2.6.0
- torchvision==0.21.0
- h5py==3.13.0
- nltk==3.9.1
- tqdm==4.67.1
- imageio==2.37.0
- numpy==2.2.4
- pillow==11.1.0
- matplotlib==3.10.1
- scikit-image==0.25.2

## Installation

- Set up the environment and dependencies using one of the following methods:
  - using pip: ``pip install -r requirements.txt``
  - using conda: install the attached *.env by running ``conda create env -f cv-g10-env.yml ``then activate the installed enviroment by running ``conda activate cv-g10-env``
- Down the [training data](https://drive.google.com/drive/folders/17zemySCBHobht_r-HvfN_ak8Ppt8VpaR) and put it in a folder named `_data`
- If you don't want to train it you can just down our [trained model](https://drive.google.com/drive/folders/1oBrv7iFwWiobD7lutT948X_3S3nW3v_D?usp=sharing) and put it in a folder named `checkpoints`
- In the `scripts` folder run:
  - `train.py`: to train the model
  - `eval.py`: to evaluate the model (BLEU, Meteor)
- To see generated caption for an image see the `User Guide` section

## User Gui

Run the following command to generate captions for an image:

```bash
python caption.py -i reports/attachments/test00.jpg -m checkpoints/best_model.pth.tar -wm _data\flickr8k\WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json -b 5
```

## References

- [a PyTorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning?tab=readme-ov-file)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- [Image Captioning with Visual Attention](https://arxiv.org/abs/1502.03044)

## License

[MIT License](https://mit-license.org/)
