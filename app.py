import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings  # Use this import
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Function to read PDFs from a directory
def read_pdfs(folder):
    file_loader = PyPDFDirectoryLoader(folder)
    pdfs = file_loader.load()
    return pdfs

# Function to chunk documents
def chunk_documents(documents, chunk_size=500, chunk_overlap=0):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return chunks

# Updated Cohere Embeddings configuration
try:
    embeds = CohereEmbeddings(
        cohere_api_key=os.getenv("COHERE"),
        model="embed-english-v3.0"  # Updated model
    )

    # Read and process PDF documents
    pdfs = read_pdfs('uploads')
    chunked_pdfs = chunk_documents(documents=pdfs)

    # Create a FAISS vector store
    docsearch = FAISS.from_documents(chunked_pdfs, embeds)

    # Rest of your code remains the same...

except Exception as e:
    print(f"Error initializing embeddings: {e}")

pdfs = read_pdfs('uploads')
chunked_pdfs = chunk_documents(documents=pdfs)

# Create a FAISS vector store
docsearch = FAISS.from_documents(chunked_pdfs, embeds)

# Define the prompt template
prompt_template = """Text: {context}

Question: {question}

Answer the question based on the PDF Document provided. If the text doesn't contain the answer, reply that the answer is not available.
Do Not Hallucinate."""
prompt = PromptTemplate.from_template(prompt_template)

# Initialize LLM
llm = Cohere(
    model="command-nightly", 
    temperature=0.9,
    cohere_api_key=os.getenv("COHERE_API_KEY")
)


# Create retriever and processing chain
retriever = docsearch.as_retriever()
retrievable = RunnableParallel(
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
)
chain = retrievable | prompt | llm | StrOutputParser()

# Process a question
question = "How Do Computers Work?"
output = chain.invoke(question)

# Print the output
print(output)
