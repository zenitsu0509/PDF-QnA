from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

from document_processor import PDFProcessor
from vectorstore import VectorStore
from rag_engine import RAGEngine

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global vector store and RAG engine
vector_store = VectorStore()
rag_engine = RAGEngine(vector_store)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process PDF
        document_chunks = PDFProcessor.extract_text_from_pdf(filepath)
        vector_store.add_documents(document_chunks)
        
        return jsonify({'message': 'PDF uploaded and processed successfully'}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    query = request.json.get('question', '')
    
    if not query:
        return jsonify({'error': 'No question provided'}), 400
    
    answer = rag_engine.answer_question(query)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
