import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Optional

load_dotenv()

class PDFQuestionAnswerer:
    def __init__(self, embedding_model: str = "embed-english-v3.0", 
                 llm_model: str = "command-nightly"):
        """
        Initialize the PDF Question Answerer
        
        :param embedding_model: Cohere embedding model to use
        :param llm_model: Cohere language model to use
        """
        # Initialize Cohere Embeddings
        self.embeds = CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model=embedding_model
        )
        
        # Initialize Cohere LLM
        self.llm = Cohere(
            model=llm_model, 
            temperature=0.9,
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )
        
        # Initialize vector store and chain as None
        self.docsearch = None
        self.chain = None
        self.current_pdf_path: Optional[str] = None
    
    def load_and_process_pdf(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 0):
        """
        Load and process a PDF file
        
        :param pdf_path: Path to the PDF file
        :param chunk_size: Size of text chunks
        :param chunk_overlap: Overlap between chunks
        """
        # Validate PDF path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Load PDF
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        except Exception as e:
            raise ValueError(f"Error loading PDF: {e}")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunked_docs = text_splitter.split_documents(documents)
        
        # Create FAISS vector store
        self.docsearch = FAISS.from_documents(chunked_docs, self.embeds)
        
        # Create retrieval chain
        retriever = self.docsearch.as_retriever()
        retrievable = RunnableParallel(
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
        )
        
        # Define prompt template
        prompt_template = """Text: {context}

Question: {question}

Answer the question based on the PDF Document provided. If the text doesn't contain the answer, reply that the answer is not available.
Do Not Hallucinate."""
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create chain
        self.chain = retrievable | prompt | self.llm | StrOutputParser()
        
        # Store current PDF path
        self.current_pdf_path = pdf_path
        
        return len(chunked_docs)  # Return number of document chunks
    
    def ask_question(self, question: str) -> str:
        """
        Ask a question about the loaded PDF
        
        :param question: Question to ask
        :return: Answer to the question
        """
        if self.chain is None:
            raise ValueError("No PDF has been loaded. Use load_and_process_pdf() first.")
        
        return self.chain.invoke(question)