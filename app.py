import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# Hugging Face API details
HF_API_URL = "https://api-inference.huggingface.co/models/mistral-ai/mistral-7b-v0"
HF_API_KEY = "YOUR_HUGGINGFACE_API_KEY"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Initialize SentenceTransformer model for embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Function to query Hugging Face API
def query_hf_api(context, question):
    payload = {"inputs": {"context": context, "question": question}}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    return response.json()

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Build FAISS index
def build_faiss_index(text_chunks):
    embeddings = embedder.encode(text_chunks, convert_to_tensor=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

# Retrieve most relevant chunks
def retrieve_relevant_chunks(question, text_chunks, index, top_k=3):
    question_embedding = embedder.encode([question], convert_to_tensor=False)
    distances, indices = index.search(np.array(question_embedding), top_k)
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    return " ".join(relevant_chunks)

# Streamlit app
st.title("PDF Question Answering with FAISS")
st.subheader("Upload a PDF and ask questions based on its content")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("Text extracted from PDF!")

    if st.checkbox("Show extracted text"):
        st.text_area("Extracted Text", pdf_text, height=300)

    # Split text into chunks
    chunk_size = 500  # Adjust based on your needs
    text_chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]

    with st.spinner("Building FAISS index..."):
        index, _ = build_faiss_index(text_chunks)
    st.success("FAISS index built!")

    question = st.text_input("Enter your question")
    if st.button("Get Answer"):
        if question.strip() == "":
            st.error("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant chunks..."):
                context = retrieve_relevant_chunks(question, text_chunks, index)
            
            with st.spinner("Generating answer..."):
                result = query_hf_api(context, question)
                answer = result.get("answer", "No answer found.")
            
            st.success("Answer Generated!")
            st.write(f"**Answer:** {answer}")
