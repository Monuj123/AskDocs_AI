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
