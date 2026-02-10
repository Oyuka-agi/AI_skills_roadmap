# PDF Text Extractor - Frontend + Backend

A simple web application to upload PDF files and extract text to CSV.

## Features
- 📤 Drag & drop or click to upload PDF
- 📄 Extract all text from PDF
- 💾 Download as CSV file
- ✨ Clean, modern UI
- 🚀 Fast processing

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Backend Server
```bash
python backend_app.py
```

The server will start at `http://localhost:8000`

### 3. Open Frontend
Open `frontend.html` in your web browser

**Or** serve it with a simple HTTP server:
```bash
# Python 3
python -m http.server 8000

# Then open http://localhost:8000/frontend.html
```

## Usage

1. Open the frontend in your browser
2. Click or drag a PDF file to the upload area
3. Click "Extract Text" button
4. CSV file will automatically download

## CSV Output Format

| Column | Description |
|--------|-------------|
| filename | Original PDF filename (without extension) |
| extracted_text | Complete text extracted from PDF |

## File Limits
- Maximum file size: 16MB
- Accepted formats: PDF only

## Tech Stack
- **Backend**: Flask + PyPDF2 + Pandas
- **Frontend**: HTML + CSS + Vanilla JavaScript
- **CORS**: Enabled for local development

## Troubleshooting

**CORS Error:**
- Make sure Flask-CORS is installed
- Backend must be running on port 5000

**File Upload Failed:**
- Check file size (max 16MB)
- Ensure file is a valid PDF

**Port Already in Use:**
```bash
# Change port in backend_app.py, line:
app.run(debug=True, port=5001)  # Use different port
```

## Project Structure
```
.
├── backend_app.py      # Flask server
├── frontend.html       # Web interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```
