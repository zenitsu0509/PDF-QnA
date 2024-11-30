import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize vector store with embedding model
        
        Args:
            model_name (str): Sentence transformer model name
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
    
    def add_documents(self, documents):
        """
        Add documents to vector store
        
        Args:
            documents (List[Dict]): List of document chunks
        """
        # Generate embeddings
        embeddings = self.embedding_model.encode([doc['text'] for doc in documents])
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Store original documents
        self.documents = documents
    
    def search(self, query, top_k=3):
        """
        Search for most relevant documents
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
        
        Returns:
            List of most relevant documents
        """
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        
        return [self.documents[i] for i in indices[0]]