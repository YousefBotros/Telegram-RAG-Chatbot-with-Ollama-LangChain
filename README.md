# 🤖 Telegram RAG Chatbot with Ollama

A production-ready Telegram bot that uses **RAG (Retrieval-Augmented Generation)** to answer questions based on your knowledge base. Powered by Ollama LLMs, LangChain, and ChromaDB.

## ✨ Features

- 🧠 **RAG Pipeline** - Retrieves relevant context from documents
- 📚 **Document Support** - Upload PDF/TXT files to knowledge base
- 🔄 **Conversation Memory** - Remembers chat context
- 🌍 **Multi-language** - Supports English and Arabic
- 📎 **File Processing** - Automatic document ingestion
- ⚡ **Fast Response** - Optimized with FAISS/ChromaDB

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Ollama (if not installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.1

# Start Ollama service
ollama serve
