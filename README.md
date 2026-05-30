# 3D Shape Classification with Multi-View Images

A PyTorch project for classifying 3D objects from multiple rendered grayscale views.  
The project implements and compares two architectures: a lightweight CNN with depthwise separable convolutions and a Tiny Vision Transformer (TinyViT). Each 3D shape is represented by several 2D rendered views, and the final object prediction is produced using multi-view pooling.

## Short Description

This repository contains a multi-view 3D shape classification system built with PyTorch. It trains image-based classifiers on rendered views of 3D objects and evaluates the final shape category using mean or max pooling across the views of each object. The project includes data preprocessing utilities, model definitions, training, testing, checkpoint saving, and per-category accuracy reporting.


## Dataset

The dataset is **not included** in this repository because it contains many files and is too large for normal GitHub storage.

Expected dataset format:

```text
dataset/
├── train/
│   ├── category_1/
│   │   ├── shape_1/
│   │   │   ├── shape_1_001.png
│   │   │   ├── shape_1_002.png
│   │   │   └── ...
│   │   └── shape_2/
│   └── category_2/
└── test/
    ├── category_1/
    └── category_2/
```

Each object belongs to a category folder, and each shape folder contains multiple rendered grayscale PNG views.

## How to Handle the Dataset

Since the dataset is large, do **not** push it directly to GitHub. Recommended options:


## Training

The training process:

1. Loads all rendered PNG images.
2. Converts them to grayscale.
3. Normalizes the image values.
4. Saves preprocessed arrays as compressed `.npy.gz` files.
5. Trains the selected model.
6. Saves checkpoints inside the `model/` folder.
7. Saves the training/validation loss and error plot as `errorplot.png`.

## Models

### CNN

The CNN is designed to be lightweight and parameter-efficient. It uses:

- Standard convolution
- Batch normalization
- Leaky ReLU
- Max pooling
- Depthwise separable convolutions
- Adaptive average pooling
- 1×1 convolution as the classification head

### TinyViT

The Tiny Vision Transformer uses:

- Patch embeddings
- Learnable positional embeddings
- Learnable class token
- Transformer encoder layers
- Layer normalization
- Linear classification head

## Evaluation

The testing script evaluates each shape using all available views. It supports two pooling strategies:

- **Mean pooling:** averages class probabilities across all views.
- **Max pooling:** keeps the strongest class probability across all views.

The output includes:

- Per-category accuracy
- Overall test error
- Overall test accuracy

## Results

In the report, the TinyViT model achieved the best result with mean pooling:

```text
Overall Test Error: 14.00%
Accuracy: 86.00%
```

The CNN also performed well, especially with mean pooling:

```text
Overall Test Error: 18.00%
Accuracy: 82.00%
```

## Course Info

Assignment based on **3D Shape Classification** for **Computer Vision (INF417)**.
