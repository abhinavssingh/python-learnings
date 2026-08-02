# Capstone Session 12 Requirements

# Project Title
Enhancing Dental X-rays Using Autoencoders

## Business Objective
Build an Autoencoder-based Deep Learning solution to denoise panoramic dental X-ray images and improve image clarity.

## Dataset
File:
- Dental-Panaromic-Autoencoder.npz

Dataset contains:
- x_train
- y_train
- x_test
- y_test

The dataset consists of anonymized panoramic dental X-rays from 116 patients and is provided as compressed NumPy arrays. citeturn112search1

## Task A

### Dataset Loading
- Load dataset using numpy.load().
- Extract:
  - x_train
  - y_train
  - x_test
  - y_test

### Noise Generation
Create noisy images.

Parameters:
- noise_factor = 0.2

Example:
- x_train_noisy = x_train + noise_factor * random_noise
- x_test_noisy = x_test + noise_factor * random_noise

Requirements:
- Use random values from normal distribution.
- Clip values between 0 and 1. citeturn112search1

### Visualization
Plot:
- First 5 original X-ray images from x_train
- First 5 noisy X-ray images from x_train_noisy citeturn112search1

## Autoencoder Model

### Input Shape
- 256 x 256 x 3

### Create Denoise Class
- Inherit from keras.Model

### Encoder Architecture
1. Input Layer
   - Shape = (256,256,3)
2. Conv2D
   - Filters = 64
   - Kernel Size = (3,3)
   - Activation = relu
   - Padding = same
   - Strides = 2
3. Conv2D
   - Filters = 32
   - Kernel Size = (3,3)
   - Activation = relu
   - Padding = same
   - Strides = 2

### Decoder Architecture
1. Conv2DTranspose
   - Filters = 32
   - Kernel Size = (3,3)
   - Activation = relu
   - Padding = same
   - Strides = 2
2. Conv2DTranspose
   - Filters = 64
   - Kernel Size = (3,3)
   - Activation = relu
   - Padding = same
   - Strides = 2
3. Conv2D
   - Filters = 1
   - Kernel Size = (3,3)
   - Activation = sigmoid
   - Padding = same

### Forward Pass
Implement call() method:
- Pass input to encoder
- Pass encoded output to decoder
- Return reconstructed image

## Compilation
Optimizer:
- Adam

Loss Function:
- MeanSquaredError (MSE) citeturn112search1

## Training
Input:
- X = x_train_noisy

Target:
- Y = x_train

Validation:
- Validation X = x_test_noisy
- Validation Y = x_test

Epochs:
- 50

## Training Visualizations
Plot:
- Training Loss vs Epochs
- Validation Loss vs Epochs
- Training MAE vs Epochs
- Validation MAE vs Epochs citeturn112search1

## Task B

### Model Evaluation
- Evaluate autoencoder using x_test.

### Reconstruction
Process:
1. Pass x_test into encoder.
2. Generate encoded representation.
3. Pass encoded output into decoder.
4. Generate reconstructed images.

### Results Visualization
Plot:
- First 10 noisy images from x_test_noisy.
- First 10 denoised/reconstructed images produced by autoencoder.

Compare original and reconstructed outputs to evaluate denoising performance. citeturn112search1

## Deliverables

### Data Preparation
- NPZ loading
- Noise generation
- Data validation

### Visualizations
- Original images
- Noisy images
- Training curves
- Denoised images

### Deep Learning Model
- Denoise Autoencoder class
- Encoder
- Decoder
- call() implementation

### Evaluation
- Reconstruction quality
- Loss
- MAE
- Denoised image comparison

## Final Outcome
Build a convolutional autoencoder capable of learning image representations and reconstructing cleaner panoramic dental X-rays from noisy inputs. 
