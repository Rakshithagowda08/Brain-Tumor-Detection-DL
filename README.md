# Brain-Tumor-Detection-DL
# Brain Tumor MRI Classification and Explainability System

## Overview

This project presents an end-to-end deep learning–based system for automated brain tumor classification from MRI images with a strong emphasis on interpretability and clinical transparency. The system integrates two state-of-the-art convolutional neural networks, ResNet50 and EfficientNet-B0, to classify MRI scans into four categories: glioma, meningioma, pituitary tumor, and no tumor.

To enhance trust and usability in medical decision support, the framework incorporates Grad-CAM–based visual explanations, confidence interpretation, and automated medical-style PDF report generation. The final system is deployed as an interactive web application using Streamlit.

---

## Key Features

- Dual-model architecture (ResNet50 + EfficientNet-B0)
- MRI-based brain tumor classification
- Grad-CAM attention map visualization
- Confidence calibration and explanation
- Side-by-side model comparison per image
- Downloadable Grad-CAM images
- Automated PDF medical-style report generation
- Deployment-ready Streamlit web application

---

## Tumor Classes

The system classifies MRI images into the following categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## System Architecture

1. **Input**: Single MRI image uploaded via web interface  
2. **Preprocessing**: Resizing, normalization  
3. **Prediction**:
   - ResNet50 classifier
   - EfficientNet-B0 classifier  
4. **Explainability**:
   - Grad-CAM heatmap generation
   - Attention map legend  
5. **Output**:
   - Predicted class and confidence
   - Visual explanation
   - Downloadable report and figures  

---

## Dataset Structure

The dataset must be organized as follows:

Training/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/

Testing/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/


Each folder contains MRI images corresponding to its class.

---

## Model Training (Optional)

If you wish to retrain the models with additional data:

1. Add new MRI images to the appropriate class folders
2. Open the training notebook:Brain_Tumor_Model.ipynb
3. Train the models
4. Save updated models as:
- `resnet50_brain_tumor.h5`
- `efficientnet_b0_brain_tumor.h5`

---

## Running the Application Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt

Run Streamlit App
streamlit run app.py


# Using the Web Application

Open the web interface

Upload a brain MRI image (.jpg, .png, .jpeg)

View predictions from both models

Examine Grad-CAM visual explanations


#Download

Grad-CAM image
PDF Medical Report

# The generated report includes

Predicted tumor type
Confidence score
Model reliability explanation
Grad-CAM visualization
Attention map legend
Clinical interpretation
Disclaimer for AI-assisted analysis

This report is designed to resemble professional clinical documentation for demonstration and academic purposes.

# Explainability and Trust

Grad-CAM visualizations highlight regions of the MRI that most influenced the model’s prediction. This improves transparency and helps users understand model behavior rather than relying on black-box outputs.

# Deployment Options

# The application is compatible with:

Streamlit Cloud
Hugging Face Spaces
Local or institutional servers

Deployment requires only the trained models and application files.

# Ethical Disclaimer

This system is intended for research and educational purposes only. It is not a replacement for professional medical diagnosis. Clinical decisions must always be validated by qualified healthcare professionals.

# Future Enhancements

Multi-sequence MRI support (T1, T2, FLAIR)
Confidence calibration techniques
Ensemble decision strategies
Patient-wise longitudinal analysis
Clinical validation with expert annotations

# Author

Developed as an academic and research-oriented project in medical image analysis and explainable artificial intelligence.

# License

This project is released for academic and non-commercial use.


---


