# Male-Female_Classification_Project

# Men Women Classification

![Men Women Classification](<Add a link to an example image or a logo if available>)

This project focuses on classifying images of men and women using a Convolutional Neural Network (CNN). The dataset used for training and evaluation is sourced from [Kaggle](https://www.kaggle.com/saadpd/menwomen-classification).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Getting Started](#getting-started)
- [License](#license)

## Overview

Men Women Classification is a deep learning project designed to classify images into two classes: 'men' and 'women.' The project uses a Convolutional Neural Network (CNN) to achieve accurate classification results.

## Dataset

The dataset consists of images categorized into two classes: 'men' and 'women.' After downloading the dataset, it was preprocessed and transformed to create a PyTorch `ImageFolder` dataset.

## Model Architecture

The neural network model used for classification is a simple CNN created with PyTorch's `nn.Sequential`. The architecture includes convolutional layers with ReLU activation functions, max-pooling layers, and fully connected layers.

```python
class class_finder(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # ... More convolutional layers ...

            # Fully connected layers
            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, xb):
        return self.network(xb)
