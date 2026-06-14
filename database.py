"""
Vector database management for RAG
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from config import config

class VectorDatabase:
    def __init__(self):
        """Initialize vector database with embeddings"""
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vectorstore = None
        self._load_or_create_db()
    
    def _load_or_create_db(self):
        """Load existing database or create new one"""
        if os.path.exists(config.PERSIST_DIR):
            print(f"Loading existing database from {config.PERSIST_DIR}")
            self.vectorstore = Chroma(
                persist_directory=config.PERSIST_DIR,
                embedding_function=self.embeddings
            )
        else:
            print("No existing database found. Creating new one...")
            self._create_vectorstore()
    
    def _create_vectorstore(self):
        """Create vectorstore from knowledge base documents"""
        documents = []
        
        if not os.path.exists(config.KNOWLEDGE_BASE_DIR):
            os.makedirs(config.KNOWLEDGE_BASE_DIR)
            print(f"Created {config.KNOWLEDGE_BASE_DIR}. Please add documents there.")
            return
        
        # Load all documents from knowledge_base directory
        for file in os.listdir(config.KNOWLEDGE_BASE_DIR):
            file_path = os.path.join(config.KNOWLEDGE_BASE_DIR, file)
            
            try:
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded PDF: {file}")
                    
                elif file.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded TXT: {file}")
                    
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if documents:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks from {len(documents)} documents")
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=config.PERSIST_DIR
            )
            self.vectorstore.persist()
            print("Vector database created and persisted!")
        else:
            print("No documents found. Add PDF or TXT files to ./data/knowledge_base/")
    
    def add_document(self, file_path):
        """Add a new document to the vector database"""
        if not self.vectorstore:
            self._create_vectorstore()
        
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError("Only PDF and TXT files are supported")
        
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        
        return len(chunks)
    
    def search(self, query, k=config.TOP_K_RESULTS):
        """Search for relevant documents"""
        if not self.vectorstore:
            return []
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def get_relevant_context(self, query):
        """Get relevant context as formatted string"""
        results = self.search(query)
        
        if not results:
            return ""
        
        context_parts = []
        for doc, score in results:
            if score > 0.5:  # Only include relevant results
                context_parts.append(doc.page_content)
        
        return "\n\n".join(context_parts)

# Singleton instance
db = VectorDatabase()
