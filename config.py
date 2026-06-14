"""
Configuration settings for Telegram RAG Bot
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Telegram
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")
    
    # Ollama
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1")
    
    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # RAG Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 4
    
    # Database
    PERSIST_DIR = "./chroma_db"
    
    # File paths
    KNOWLEDGE_BASE_DIR = "./data/knowledge_base"
    
    # Bot settings
    MAX_HISTORY = 10  # Max conversation history per user
    MAX_MESSAGE_LENGTH = 4000

config = Config()
