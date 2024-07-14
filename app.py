import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import PyPDF2

# Load the model and tokenizer
output_dir = './ResumeClassifier_model/'
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)

# Load label encoder classes
label_encoder = LabelEncoder()
label_classes_path = os.path.join(output_dir, 'label_classes.npy')
if os.path.exists(label_classes_path):
    label_encoder.classes_ = np.load(label_classes_path, allow_pickle=True)
else:
    raise FileNotFoundError(
        f"The file {label_classes_path} does not exist. Ensure you have saved the label encoder classes.")

# Use GPU if available
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# Function to extract text from PDF


def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to preprocess the text


def preprocess_text(text, tokenizer, max_len=256):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoded['input_ids'], encoded['attention_mask']

# Function to predict job category


def predict_job_category(text, model, tokenizer, label_encoder):
    inputs, masks = preprocess_text(text, tokenizer)
    inputs, masks = inputs.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(inputs, attention_mask=masks)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).cpu().numpy()
    predicted_label = label_encoder.inverse_transform(prediction)

    return predicted_label[0]

# Streamlit app


def main():
    st.title('Resume Job Category Prediction')

    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

    if uploaded_file is not None:
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(uploaded_file)

            # Predict job category
            predicted_category = predict_job_category(
                text, model, tokenizer, label_encoder)

            st.write(f'The predicted job category is: {predicted_category}')

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == '__main__':
    main()
