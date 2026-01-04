# Pneumonia Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)


A comprehensive deep learning pipeline for automated pneumonia detection from chest X-ray images using convolutional neural networks and transfer learning with ResNet50 and EfficientNetB0 architectures. The project covers the complete workflow from dataset analysis through model training, evaluation, and clinical interpretation.

## Live Demo

**Try it now:** [pneumoai.vercel.app](https://pneumoai.vercel.app)

Experience the power of AI-driven pneumonia detection through our user-friendly web interface. Upload chest X-ray images and receive instant AI-powered diagnostics using our ResNet50 model trained on thousands of medical images. The web application provides fast, accurate predictions with a clean, intuitive interface designed for both medical professionals and researchers.

<p align="center">
  <img src="assets/pneumoai-screenshot.png" alt="PneumoAI Web Application" width="800">
</p>

*AI-powered chest X-ray analysis platform featuring ResNet50 architecture with high accuracy and fast inference capabilities.*

## Overview

This project implements a complete medical image classification system designed to assist in the detection of pneumonia from chest X-ray images. The system employs state-of-the-art deep learning techniques including transfer learning from ImageNet-pretrained models, medical-appropriate data augmentation, and comprehensive clinical evaluation metrics.

**Dataset:** [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle

The dataset contains chest X-ray images organized into two categories: NORMAL and PNEUMONIA. Images are preprocessed, analyzed for distribution, and split into training, validation, and test sets using stratified sampling to maintain class proportions.

## Workflow

### 1. Setup and Configuration
- Imports required libraries: TensorFlow, Keras, scikit-learn, matplotlib, seaborn
- Configures GPU memory growth for efficient training
- Sets random seeds for reproducibility
- Defines constants: image size (224x224), batch size (32), and training parameters

### 2. Dataset Preparation
- Downloads the dataset using the Kaggle API via `kagglehub`
- Analyzes class distribution and image properties
- Creates stratified train/validation/test splits (80/10/10 ratio)
- Visualizes sample images from both classes
- Computes class weights to handle imbalanced data

### 3. Data Augmentation
Applies conservative augmentations suitable for medical images:
- Rotation (up to 20 degrees)
- Width and height shifts (up to 10%)
- Zoom (up to 10%)
- Brightness adjustments
- Horizontal flips

### 4. Model Architecture

**Baseline CNN (from scratch)**
- 4 convolutional blocks with batch normalization
- MaxPooling layers for spatial dimension reduction
- Dense layers with dropout for regularization
- Binary classification output

**ResNet50 (Transfer Learning)**
- Pre-trained on ImageNet with frozen base layers
- Grayscale to RGB conversion (3-channel input)
- Two-phase training approach:
  - Phase 1: Train only the top classification layers
  - Phase 2: Fine-tune the last 30 layers of the base model
- Global average pooling and dense classification head

**EfficientNetB0 (Transfer Learning)**
- Similar architecture to ResNet50 pipeline
- More efficient parameter usage
- Two-phase training with fine-tuning

### 5. Training Strategy
- Early stopping with patience to prevent overfitting
- Learning rate reduction on plateau for better convergence
- Model checkpointing to save best weights based on validation loss
- Binary cross-entropy loss function
- Adam optimizer with initial learning rate adjustments

### 6. Evaluation Metrics

**Classification Metrics:**
- Accuracy: Overall correctness of predictions
- Precision (PPV): Positive predictive value
- Recall (Sensitivity): True positive rate
- Specificity: True negative rate
- F1-Score: Harmonic mean of precision and recall
- AUC-ROC: Area under the receiver operating characteristic curve

**Clinical Interpretation:**
- Confusion matrix analysis
- True/false positive and negative rates
- Clinical significance of misclassifications

### 7. Model Comparison
All three models are evaluated on the validation set, and the model with the highest AUC score is selected for final testing. The comparison includes validation performance metrics and training curves.

## Results

**Best Model:** ResNet50 with fine-tuning

The final model achieves strong performance on the test set with clinically relevant metrics. The confusion matrix provides insights into the model's ability to correctly classify pneumonia cases and normal X-rays, with special attention to false negatives (missed pneumonia cases) which are critical in medical diagnosis.

All results, including trained models, training history, confusion matrices, ROC curves, and performance metrics, are saved to the `models/` directory for future reference and deployment.

## Installation

Install the required dependencies:

```bash
pip install tensorflow keras scikit-learn matplotlib seaborn pillow opencv-python kagglehub
```

## Usage

1. Ensure you have Kaggle API credentials configured (`~/.kaggle/kaggle.json`)
2. Open and run the Jupyter notebook: `pneumonia-detection-using-resnet.ipynb`
3. The notebook will automatically download the dataset and train all models
4. Trained models and results will be saved to the `models/` directory

## Project Structure

```
.
├── pneumonia-detection-using-resnet.ipynb  # Main notebook
├── README.md                                # Project documentation
└── models/                                  # Generated during training
    ├── baseline_cnn_best.keras             # Baseline CNN model
    ├── resnet50_best.keras                 # ResNet50 model
    ├── efficientnet_best.keras             # EfficientNetB0 model
    ├── training_history.pkl                # Training history data
    ├── results_summary.json                # Performance metrics
    └── *.png                               # Visualization plots
```

## Technical Details

**Image Preprocessing:**
- Input size: 224x224 pixels
- Normalization: Pixel values scaled to [0, 1]
- Grayscale to RGB conversion for transfer learning models

**Training Configuration:**
- Batch size: 32
- Image size: 224x224
- Early stopping patience: 10 epochs
- Learning rate reduction factor: 0.5
- Random seed: 42 (for reproducibility)

## Future Enhancements

- Implement Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explanations of model predictions
- Validate performance on external chest X-ray datasets from different sources
- Collaborate with medical professionals for clinical validation and deployment considerations
- Optimize model for edge deployment on mobile or embedded devices

## Acknowledgements

- Dataset provided by Paul Mooney via the [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Built with TensorFlow, Keras, and scikit-learn
- Transfer learning based on ImageNet pre-trained weights

**Heshan Ranasinghe**  
Electronic and Telecommunication Engineering Undergraduate

- Email: hranasinghe505@gmail.com
- GitHub: [@DPHeshanRanasinghe](https://github.com/DPHeshanRanasinghe)
- LinkedIn: [Heshan Ranasinghe](https://www.linkedin.com/in/heshan-ranasinghe-988b00290)
