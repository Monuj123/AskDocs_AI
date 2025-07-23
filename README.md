# 📚 Document Question Answering System

A document-based question answering system using vector embeddings and retrieval-augmented generation powered by OpenAI and ChromaDB.

## ✨ Features

- 📂 Document ingestion from directory
- ✂️ Smart text chunking with overlap
- 🔍 Semantic search with ChromaDB
- 🤖 AI-powered answers using GPT-3.5
- 💾 Persistent vector storage
- 🔧 Configurable chunking parameters

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/document-qa-system.git
   cd document-qa-system
  
2. Set up virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows   
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
6. Create .env file:
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env

### Usage
1. Add your text documents to ./news_articles directory
2. Run the system:
   ```bash
   python main.py
3. Enter your questions when prompted

## 🛠 Configuration
Modify these in main.py:
   ```bash
   # Document processing
   CHUNK_SIZE = 1000      # Characters per chunk
   CHUNK_OVERLAP = 20     # Overlap between chunks
   
   # Query settings
   N_RESULTS = 2          # Number of chunks to retrieve
   
   # Model settings
   EMBEDDING_MODEL = "text-embedding-3-small"
   LLM_MODEL = "gpt-3.5-turbo"
   

