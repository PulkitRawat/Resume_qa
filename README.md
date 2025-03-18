# Resume QnA

## Overview
Resume QnA is a Retrieval-Augmented Generation (RAG) model designed to extract relevant information from resumes uploaded as PDFs or images. The system utilizes fine-tuned BERT transformer-based embeddings to find the closest sentences to a given question and generate a concise response. Fine-tuning has been performed on a resume-specific dataset to enhance answer accuracy. The application is deployed using Flask, providing a simple web interface for users to upload documents and ask questions.

## Features
- Supports both scanned and unscanned PDFs.
- Extracts text from uploaded resumes and finds relevant answers.
- Uses fine-tuned BERT-based sentence embeddings for efficient retrieval.
- Fine-tuned QnA model for generating accurate responses.
- Simple web-based interface for easy interaction.
- Flask-based backend for deployment.

## Usage
### 1. Cloning the repo
Clone the repo and the change the directory to the app directory:
```bash
git clone https://github.com/PulkitRawat/Resume_qa.git
cd Resume_qa
```
### 2. Setting Up the Environment
Ensure you have Python installed and set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```
### 3. Fine-Tuning the Embedding Model
Fine-tune the embedding model using `embedding_fine_tune.ipynb` and the provided dataset:
- Place training, validation, and test data inside respective folders.
- Run `embedding_fine_tune.ipynb` to train the model.
- Save the fine-tuned model and update `model.py` accordingly.

### 4. Fine-Tuning the QnA Model
Fine-tune the QnA model using `model_fine_tune.ipynb` and the `qca` dataset:
- Load the dataset into the notebook.
- Train the model using the provided scripts.
- Save the updated model and integrate it with the main pipeline

### 5. Running the Web Application
Start the Flask server:
```bash
python deployment.py
```
Open `http://127.0.0.1:5000/` in your browser to interact with the application.



## Structure
```
PDF-Answering-AI/
│── templates/
│   ├── index.html            # Upload and question input form
│   ├── result.html           # Displays the generated answer
│── deployment.py             # Flask app for serving the model
│── model.py                  # Core model logic (text extraction, embeddings, QnA inference)
│── embedding_fine_tune.ipynb # Notebook for fine-tuning embedding model
│── model_fine_tune.ipynb     # Notebook for fine-tuning QnA model
│── qca/                      # Dataset for fine-tuning QnA model
│── train/                    # Training data for embedding model
│── validation/               # Validation data for embedding model
│── test/                     # Test data for embedding model
│── pdf_extractor.ipynb       # reference to change/test pdf extraction step
│── qa.ipynb                  # reference to test/change the code in notebook
```

## Conclusion
This project provides an efficient, domain-specific AI assistant for answering questions from resumes. Ensure proper fine-tuning and model integration for optimal results.

