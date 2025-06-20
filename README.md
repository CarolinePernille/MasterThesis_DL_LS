# Master's Thesis: Deep Learning on Magnetic Resonance k-space Data to Improve Image Quality

This repository contains code and utilities for image reconstruction using deep learning models. It includes the necessary 
preprocessing steps, as well as training and evaluating models for both 3D and 4D reconstruction.

## Table of Contents
- [Overview](#overview)
- [Preprocessing](#preprocessing)
- [3D-models](#3d-models)
- [4D-models](#4d-models)
- [Dependencies](#dependencies)
- [Usage](#usage)


## Overview
The aim of this thesis was to use deep learning to improve the image quality of Looping Star images with the temporal dimension included, using k-space data as input data. The proposed pipeline is split in two; a 3D-UNet to reconstruct
3D Looping Star images from 3D k-space, and a 4D-UNet to improve the image quality of the reconstructed Looping Star images, with the temporal dimension included.

## Preprocessing
The preprocessing folder contain mutliple jupyter notebooks and python utilites for preprocessing data in preparation for deep learning. 

### Contents
- preprocess_kspace.ipynb: Preprocessing of gridded k-space data. Includes transformation to image space, coil combination, cropping in image space, and assembling 3D volumes into 4D volumes.
- 3D_preprocessing.ipynb: Division of training, test and validation data and preprocessing steps for training of the 3D-UNet (including normalization, scaling and splitting 4D volumes into 3D volumes)
- 4D_preprocessing.ipynb: Initializing the 3D-UNet with weights from the best performing models and assembling the predictions for each participant into 4D volumes for training of the 4D-UNet. Includes also normalization and scaling of the ground truth labels for the 4D-UNet.
- preprocessing_utils.py: Utilities used for both 3D and 4D preprocessing.

## 3D-models
The 3D models folder contain the model training, model selection and model evaluation of the 3D-UNet.

### Contents
- 3D_UNet_complex.ipynb: The training process of the 3D-UNet. Cell outputs are included for the training of 3 of the 12 models.
- 3D_model_selection.ipynb: The model selection process and training evaluation of the 3D-UNet.
- 3D_model_evaluation.ipynb: The model and result evaluation on test data.
- train_utils.py: Utilities used for model training, model selection and model evaluation.

## 4D-models
The 4D models folder contain the developed 4D-UNet and the model training, model selection and model evaluation of the 4D-UNet.

### Contents
- models.py: Script containing the developed 4D-UNet. Also containing a more complex version of the 4D-UNet not used in this project.
- 4D_UNet.ipynb: The training process of the 4D-UNet. Cell outputs are included for the training of 3 of the 10 models.
- 4D_model_selection.ipynb: The model selection process and training evaluation of the 4D-UNet.
- 4D_model_evaluation.ipynb: The model and result evaluation on test data.
- 4D_SNR.ipynb: ROI placement determination and tSNR calculation for both EPI, reconstructed LS and original LS images.
- train_utils.py: Utilities used for model training, model selection and model evaluation.

## Dependencies
To run the code in this repository, you will need the following dependencies:
- Python (version 3.11.10)
- PyTorch (version 2.5.1)
- pytorch-3dunet (version 1.9.1 - https://github.com/wolny/pytorch-3dunet)
- convNd (https://github.com/pvjosue/pytorch_convNd/tree/master)
- Nibabel (version 5.3.2)
- torchmetrics (version 1.6.1)
- matplotlib (version 3.9.2)
- NumPy (version 1.26.4)
- h5py (version 3.12.1)
- torchmetrics (version 1.6.1)
- scipy (version 1.13.1)
- ignite (version 0.5.1)

## Usage
The scrips in this repository compose a deep learning pipeline used for this master's thesis. By installing the required dependancies above, all functions and classes should be usable. Paths and directory names will need to be updated accordingly.


