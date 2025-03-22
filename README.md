# DCGAN for Pokémon Image Generation 🎨🐾

This project implements and deploys a Deep Convolutional Generative Adversarial Network (DCGAN) to generate Pokémon-style images. The model is trained using PyTorch and deployed via a Flask web application that allows users to generate new Pokémon-like images on demand.

## 🔍 Overview

Generative Adversarial Networks (GANs) have revolutionized image synthesis. This project specifically uses a DCGAN architecture — a variant using convolutional layers for better stability and quality — to generate synthetic Pokémon images from random noise.

## 🧠 Model Architecture

### Generator
- Transposed convolutional layers
- Batch Normalization
- ReLU Activations
- Tanh Output Activation

### Discriminator
- Convolutional Layers
- LeakyReLU Activations
- Dropout for regularization
- Sigmoid Output for binary classification

## 📁 Dataset

- Pokémon sprite dataset from Kaggle (7,000 images)
- Preprocessing: Normalization to [-1, 1], Data Augmentation
- Loaded using `torchvision.datasets.ImageFolder`

## ⚙️ Training Details

- **Framework:** PyTorch
- **Optimizer:** Adam
- **Loss Function:** `BCEWithLogitsLoss`
  
- **Hyperparameters:**
  - Batch Size: 128
  - Learning Rate: 0.0002
  - Beta1: 0.5
  - Epochs: 200
  - Latent Vector Size: 100

### Training Loop
1. Train Discriminator on real and generated images
2. Train Generator to fool the Discriminator
3. Save losses and generate samples during training
4. Save checkpoints regularly

## 🚀 Flask Web App Deployment

A simple Flask web app was created to host the trained DCGAN model.

### Features:
- Web interface to generate new Pokémon-style images
- Backend serves real-time generation using the Generator model
- API endpoints for generation and retrieval


