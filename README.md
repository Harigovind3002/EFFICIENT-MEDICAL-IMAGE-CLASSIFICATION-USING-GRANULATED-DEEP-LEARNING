# Efficient Medical Image Classification using Granulated Deep Learning

## Overview
This repository contains the code for my final year capstone project. It implements a robust classification pipeline designed to evaluate **Granular Computing** across multiple medical imaging modalities and standard visual benchmarks. The project demonstrates that extracting localized textural features via granulation outperforms standard max-pooling operations for identifying fine-grained anomalies, particularly in noisy medical data.

## Methodology
The pipeline compares a standard 2D Convolutional Neural Network (CNN) baseline against two custom granular architectures:
1. **Standard CNN**: A traditional baseline model.
2. **Fixed Granules CNN**: Utilizes a custom `GranularConv2D` TensorFlow layer to extract overlapping patches (e.g., 3x3 and 5x5) before applying convolutions, preserving the spatial proximity of medical opacities and lesions.
3. **Arbitrary Granules CNN**: Implements region-growing segmentation based on Otsu thresholding to dynamically group pixels with similar intensities before convolution.

All models are rigorously evaluated using **10-fold cross-validation** to ensure statistical significance.

## Datasets Evaluated
To prove the generalization of the Granular CNN architecture, the framework was evaluated across 7 diverse datasets:

### Medical Imaging
1. **Chest X-Ray (Pneumonia):** Binary classification (Normal vs. Pneumonia) of 5,863 chest radiographs.
2. **Alzheimer’s MRI:** Multi-class classification of ~6,200 structural brain MRIs across four dementia progression stages.
3. **Medical MNIST (HeadCT Subset):** Binary/multi-class classification of brain tumors from ~3,264 grayscale axial CT scans.
4. **KVASIR (Gastrointestinal Diseases):** Multi-class classification of ~10,000 images identifying anomalies like Diverticulosis, Neoplasm, and Peritonitis.
5. **Tuberculosis X-Ray:** Binary classification for the early detection of tuberculosis conditions from radiographs.

### Standard Benchmarks
6. **MNIST:** 70,000 images for baseline 10-class handwritten digit recognition.
7. **Fashion MNIST:** 70,000 images serving as a complex visual pattern benchmark across 10 clothing categories.

## Preprocessing Pipeline
All images across all datasets were standardized using a custom OpenCV pipeline:
* Converted to Grayscale
* Resized to 128×128 pixels (or maintained at 28x28 for standard benchmarks)
* Normalized pixel values to the `[0, 1]` range
* Expanded dimensions to include a single channel for CNN compatibility
* Encoded labels numerically using `LabelEncoder`

## How to Run
1. Install the dependencies: `pip install -r requirements.txt`
2. Run the provided Jupyter Notebooks to initiate the training and 10-fold cross-validation pipelines.
