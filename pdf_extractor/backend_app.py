"""
Flask Backend for PDF Text Extractor
Handles PDF upload and returns CSV download
"""

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import pandas as pd
import os
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    reader = PdfReader(pdf_path)
    text = ""
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    return text.strip()


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF upload and return CSV"""
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text
        text = extract_text_from_pdf(filepath)
        
        # Create DataFrame
        df = pd.DataFrame({
            'filename': [os.path.splitext(filename)[0]],
            'extracted_text': [text]
        })
        
        # Convert to CSV in memory
        output = io.BytesIO()
        df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Send CSV file
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{os.path.splitext(filename)[0]}_extracted.csv'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("Starting PDF Extractor Server...")
    print("Server running at http://localhost:8000")
    app.run(debug=True, port=8000)
