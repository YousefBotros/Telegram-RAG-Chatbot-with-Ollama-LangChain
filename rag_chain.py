"""
RAG chain for retrieving and generating responses
"""

import requests
import json
from config import config
from database import db

class OllamaRAGChain:
    def __init__(self):
        self.ollama_host = config.OLLAMA_HOST
        self.model = config.MODEL_NAME
        self.system_prompt = """You are MedCare AI Assistant, a helpful healthcare assistant. Use the provided context to answer questions accurately and helpfully.

Guidelines:
1. Base your answers ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have information about that in my knowledge base"
3. Be concise and professional
4. For medical questions, always advise consulting a doctor
5. Respond in the same language as the question (English or Arabic)
6. Be empathetic and helpful

Context:
{context}

Question: {question}

Answer:"""
    
    def get_response(self, question, chat_history=None):
        """Generate response using RAG"""
        
        # Get relevant context from vector database
        context = db.get_relevant_context(question)
        
        if not context:
            context = "No specific context available. Answer based on general knowledge."
        
        # Format the prompt
        prompt = self.system_prompt.format(
            context=context,
            question=question
        )
        
        # Prepare messages for Ollama
        messages = [
            {"role": "system", "content": "You are a helpful healthcare assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history[-6:]:  # Last 3 exchanges
                messages.insert(-1, {"role": msg["role"], "content": msg["content"]})
        
        # Call Ollama API
        try:
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "Sorry, I couldn't generate a response.")
            else:
                return f"Error: Ollama returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Please make sure Ollama is running."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def simple_chat(self, question):
        """Simple chat without RAG (for general conversation)"""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": question,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response")
            return "Error generating response"
            
        except Exception as e:
            return f"Error: {str(e)}"

# Singleton instance
rag_chain = OllamaRAGChain()
