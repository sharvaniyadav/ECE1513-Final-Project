# ECE1513-Final-Project

## Aircraft Skin Defect Classification
ECE1513 – Introduction to Machine Learning 

Final Project – Fall 2025

Author: Sharvani Yadav

### 1. Overview

This project developed a machine learning system that detects and classifies aircraft skin defects from inspection images. 

The three defect categories used in this work are:

#### Crack, Missing Fastener, Paint-Off

The goal is to automate the visual inspection process to improve reliability, reduce human error, and increase maintenance efficiency.

#### Three ML solutions were implemented:

Baseline CNN (Trained from Scratch), EfficientNet-B0 Transfer Learning, Refined EfficientNet (Class Weighting + Additional Fine-Tuning)

Grad-CAM was used on the final model to improve interpretability.

### 2. Project Structure
project/
│
├── Data/                      # train, valid, test folders (3 classes)
├── baseline_cnn.pt            # saved baseline model
├── efficientnet_ft.pt         # saved EfficientNet model
├── efficientnet_refined.pt    # saved refined EfficientNet
│
├── model_comparison.csv       # final metrics summary
├── baseline_accuracy.png
├── baseline_loss.png
├── efficientnet_accuracy.png
├── efficientnet_loss.png
│
└── ECE1513_Final.ipynb        # full notebook containing all code

### 3. Dataset

The dataset used is the Roboflow Aircraft Skin Defects dataset. 

#### It contains the following directory structure:

Data/train/

Data/valid/

Data/test/


#### Training class counts:

#### Crack: 676

#### Missing-Head: 661

#### Paint-Off: 605

### 4. Models Implemented

#### Solution 1 – Baseline CNN

Simple 3-layer convolutional architecture

Trained from scratch

Basic augmentation

Achieved ~82% test accuracy

#### Solution 2 – EfficientNet Transfer Learning

Pretrained EfficientNet-B0

Trained head, then fine-tuned top layers

Improved convergence and overall generalization

Achieved ~91.5% test accuracy

#### Solution 3 – Refined EfficientNet

Added class weighting to address imbalance

Continued fine-tuning with a smaller learning rate

Achieved ~90.17% test accuracy

#### Grad-CAM Interpretability

Grad-CAM was applied to the EfficientNet model to visualize regions influencing predictions.
This confirmed that the model focuses on defect areas instead of background noise.

### 5. How to Run the Notebook
#### Step 1 — Upload Data.zip to Google Colab

!unzip Data.zip -d Data/


#### This will create:

Data/train/
Data/valid/
Data/test/

#### Step 2 — Run All Cells

The notebook automatically:

Loads and preprocesses data

Trains three models

Evaluates using the test set

Generates confusion matrices

Produces Grad-CAM heatmaps

Saves accuracy/loss plots

Saves model weights

GPU acceleration (T4 or A100) is recommended.

### 6. Results Summary

#### A comparison table is saved as model_comparison.csv.

#### Model	Test Accuracy	Macro Precision	Macro Recall	Macro F1
Baseline CNN	0.8200	0.8300	0.8200	0.8200
EfficientNet	0.9150	0.9000	0.9100	0.9050
Refined EfficientNet	0.9017	0.8993	0.8947	0.8965

EfficientNet transfer learning was the strongest performer.

### 7. Generated Files

baseline_accuracy.png

baseline_loss.png

efficientnet_accuracy.png

efficientnet_loss.png

Grad-CAM overlays (shown in notebook)

model_comparison.csv

Saved model weights (.pt files)
