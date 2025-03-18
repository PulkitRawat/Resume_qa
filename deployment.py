from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from model import * 

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles file upload, text extraction, and answer generation using the AI model.

    Steps:
    1. Validates the uploaded file and saves it securely.
    2. Extracts text from the file and generates an answer to the provided question.
    3. Renders the result page with the question and answer.

    Returns:
        Rendered HTML template with question and answer, or an error message if failed.
    """
    if 'file' not in request.files:
        return 'No file part in the request.'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file.'

    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    if not allowed_file(file.filename):
        return 'File type not allowed. Please upload a PDF or an image file (png, jpg, jpeg).'
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(file_path)

    question = request.form['question']

    try:
        pdf_text = extract_text(file_path)
        answer = find_answer(question, pdf_text)
        os.remove(file_path)
        return render_template('result.html', question=question, answer=answer)
    
    except Exception as e:
        return f"An error occurred: {e}"
    
if __name__ == '__main__':
    app.run(debug=True)