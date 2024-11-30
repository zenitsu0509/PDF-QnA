import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

class RAGEngine:
    def __init__(self, vector_store):
        """
        Initialize RAG Engine with Mistral AI API
        
        Args:
            vector_store (VectorStore): Initialized vector store
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        self.api_key = os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY must be set in .env file. Get it from https://console.mistral.ai/")
        
        # Initialize Mistral AI client
        self.client = MistralClient(api_key=self.api_key)
        
        # Store vector store for context retrieval
        self.vector_store = vector_store
        
        # Default model (can be changed if needed)
        self.model = "mistral-large-latest"
    
    def answer_question(self, query, max_tokens=300):
        """
        Answer question using RAG approach with Mistral AI
        
        Args:
            query (str): User's question
            max_tokens (int): Maximum tokens for response
        
        Returns:
            str: Generated answer with context
        """
        # Find relevant documents
        relevant_docs = self.vector_store.search(query)
        
        # Prepare context
        context = "\n\n".join([
            f"Page {doc['page']} from {doc['source']}: {doc['text']}" 
            for doc in relevant_docs
        ])
        
        # Construct comprehensive prompt
        full_prompt = f"""
        Context Information:
        {context}

        Question: {query}

        Provide a comprehensive answer based strictly on the given context. 
        If the context does not contain sufficient information to answer the question, 
        clearly state that the answer cannot be found in the provided documents.

        Answer:
        """
        
        # Prepare messages for API call
        messages = [
            ChatMessage(
                role="system", 
                content="You are a helpful assistant that answers questions based on given context. "
                        "Use only the information provided and be precise."
            ),
            ChatMessage(
                role="user", 
                content=full_prompt
            )
        ]
        
        try:
            # Generate response using Mistral AI
            chat_response = self.client.chat(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )
            
            # Extract and return the answer
            return chat_response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generating response: {str(e)}"