# Resume Classifier

This project involves the development of a resume classification system using a fine-tuned BERT model. The model classifies resumes into predefined job categories and is trained on a dataset of 2400+ resumes. The project also includes a web application for uploading resumes in PDF format and predicting the job category using Streamlit.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Streamlit Web Application](#streamlit-web-application)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Accurate classification of resumes is crucial for efficient talent acquisition and human resource management. This project leverages a fine-tuned BERT model to classify resumes into specified job categories, providing a scalable solution integrated with a user-friendly web application.

## Features

- **PDF Resume Upload**: Upload resumes in PDF format.
- **Resume Classification**: Predict job categories using a fine-tuned BERT model.
- **Web Application**: Streamlit-based interface for easy interaction.
- **Real-time Processing**: Handle resume uploads and provide instant classification results.

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dvtushar/Resume-Classifier.git
   cd resume-classifier
2. **Create a virtual environment:**:
    ```bash
    python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install the required packages:**:
     ```bash
    pip install -r requirements.txt

## Usage
To run the Streamlit web application:

1. **Activate the virtual environment:**:
   ```bash
    source venv/bin/activate  # On Windows: venv\Scripts\activate
2. **Start the Streamlit application:**:
   ```bash
   streamlit run app.py --server.enableXsrfProtection false
3. **Upload a PDF Resume:**:
   Open your browser and go to http://localhost:8501, upload a PDF resume, and view the predicted job category.

## Model Training and Evaluation

The model training process involves the following steps:
1. **Data Preparation:**:
- Convert PDFs to text using the PyPDF2 library.
- Tokenize the text data and categorize it into the 22 predefined job categories.
2. **Fine-tuning BERT:**:
- Use Hugging Face's Transformers library to fine-tune the BERT model on the dataset.
- Utilize AdamW optimizer and a learning rate scheduler.
3. **Training Loop:**:
- Train the model for 3 epochs with gradient accumulation.
- Evaluate the model on validation and test datasets.

## Results:
The model achieved the following performance metrics:
- Validation Accuracy: 0.7882037533512064
- Validation F1 Score: 0.79
- Test Accuracy: 0.7828418230563002
- Test F1 Score:  0.78
  ![image](https://github.com/user-attachments/assets/13d199a1-ca45-4339-bcf1-b819b43bd59a)
  ![image](https://github.com/user-attachments/assets/3ed70b1d-d13c-4340-bf09-0efe33d4cdcf)
  ![image](https://github.com/user-attachments/assets/6235fe0f-c2ac-4d3a-991e-864849e9ee40)

The Streamlit web application (app.py) allows users to upload resumes and get predictions for job categories.

## Screenshot of the working application
The Streamlit web application (app.py) allows users to upload resumes and get predictions for job categories.
![image](https://github.com/user-attachments/assets/568748f1-46a3-498e-bc67-ee433f13d94f)
![image](https://github.com/user-attachments/assets/31b90dda-9b73-4fd4-a238-1c48acdf5549)



