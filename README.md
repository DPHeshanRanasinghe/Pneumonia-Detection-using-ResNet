# Pneumonia Detection from Chest X-Rays

This project implements a complete deep learning pipeline for detecting pneumonia from chest X-ray images using convolutional neural networks (CNNs) and transfer learning with ResNet50 and EfficientNetB0. The workflow covers dataset analysis, preprocessing, model training, evaluation, and clinical interpretation.

## Project Highlights
- **Dataset:** Chest X-ray images from the [Kaggle Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Models:**
  - Baseline CNN (from scratch)
  - ResNet50 (transfer learning, 2-phase training)
  - EfficientNetB0 (transfer learning 2-phase training)
- **Key Features:**
  - Data analysis and visualization
  - Stratified train/validation/test split
  - Data augmentation pipeline
  - Class imbalance handling
  - Model comparison and clinical metrics

## Workflow Overview
1. **Setup & Imports:**
	- Loads all required Python libraries (TensorFlow, Keras, scikit-learn, etc.)
2. **Dataset Download & Preparation:**
	- Downloads the dataset using `kagglehub` and organizes it into train/val/test splits.
	- Analyzes and visualizes class distribution and image properties.
3. **Data Augmentation:**
	- Applies conservative augmentations (rotation, shift, zoom, brightness) suitable for medical images.
4. **Model Building:**
	- Baseline CNN: Custom architecture with 4 convolutional blocks.
	- ResNet50: Transfer learning with grayscale-to-RGB conversion, two-phase training (frozen base, then fine-tuning last 30 layers).
	- EfficientNetB0: Transfer learning with similar pipeline.
5. **Training & Evaluation:**
	- Uses early stopping, learning rate scheduling, and model checkpointing.
	- Evaluates models on test set with accuracy, precision, recall, specificity, F1-score, and AUC-ROC.
	- Visualizes confusion matrix and ROC curve.
6. **Model Comparison:**
	- Compares validation AUC of all models and selects the best (ResNet50 with fine-tuning).
7. **Clinical Metrics & Interpretation:**
	- Reports clinical metrics and interprets results for real-world relevance.

## Results Summary
- **Best Model:** ResNet50 (with fine-tuning)
- **Test Set Performance:**
  - Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, AUC-ROC
  - Confusion matrix and clinical interpretation
- **Saved Artifacts:**
  - Trained model files (`.keras`)
  - Training history (`.pkl`)
  - Results summary (`.json`)
  - Visualizations (`.png`)

## How to Run
1. Install dependencies:
	```bash
	pip install tensorflow keras scikit-learn matplotlib seaborn pillow opencv-python kagglehub
	```
2. Download the dataset (requires Kaggle API credentials).
3. Run the notebook: `Pneumonia Detection using ResNet.ipynb`

## Next Steps
- Deploy the best model as a web app (Streamlit/Gradio)
- Add Grad-CAM visualizations for explainability
- Test on external datasets
- Clinical validation

## Acknowledgements
- [Kaggle Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- TensorFlow, Keras, scikit-learn, and the open-source ML community
