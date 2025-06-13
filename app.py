# ----------------------------
# Import Statements
# ----------------------------
import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions


# ----------------------------
# Environment Configuration
# ----------------------------
# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_key = os.getenv("OPENAI_API_KEY")


# ----------------------------
# Vector Database Setup
# ----------------------------
# Initialize OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, 
    model_name="text-embedding-3-small"  # Using small model for cost efficiency
)

# Initialize persistent Chroma client with local storage
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")

# Create or get existing collection with OpenAI embeddings
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, 
    embedding_function=openai_ef
)

# Initialize OpenAI client for API calls
client = OpenAI(api_key=openai_key)


# ----------------------------
# Document Processing Functions
# ----------------------------
def load_documents_from_directory(directory_path):
    """
    Load text documents from a specified directory.
    
    Args:
        directory_path (str): Path to directory containing .txt files
        
    Returns:
        list: List of dictionaries containing document id and text content
    """
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


def split_text(text, chunk_size=1000, chunk_overlap=20):
    """
    Split text into manageable chunks with overlap for context preservation.
    
    Args:
        text (str): Input text to be chunked
        chunk_size (int): Maximum size of each chunk (default: 1000)
        chunk_overlap (int): Number of overlapping characters between chunks (default: 20)
        
    Returns:
        list: List of text chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# ----------------------------
# Data Ingestion Pipeline
# ----------------------------
# Load and process documents from directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents")

# Split documents into chunks for better embedding processing
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})


# ----------------------------
# Embedding Generation
# ----------------------------
def get_openai_embedding(text):
    """
    Generate embeddings for text using OpenAI's API.
    
    Args:
        text (str): Input text to generate embedding for
        
    Returns:
        list: Embedding vector for the input text
    """
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    print("==== Generating embeddings... ====")
    return embedding


# Generate and store embeddings for all document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_openai_embedding(doc["text"])

# Store documents with embeddings in ChromaDB
for doc in chunked_documents:
    print("==== Inserting chunks into database ====")
    collection.upsert(
        ids=[doc["id"]], 
        documents=[doc["text"]], 
        embeddings=[doc["embedding"]]
    )


# ----------------------------
# Query Processing Functions
# ----------------------------
def query_documents(question, n_results=2):
    """
    Query the document collection for relevant chunks based on a question.
    
    Args:
        question (str): The query/question to search for
        n_results (int): Number of relevant chunks to return (default: 2)
        
    Returns:
        list: Relevant document chunks that match the query
    """
    results = collection.query(query_texts=question, n_results=n_results)
    
    # Flatten the results structure
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks


def generate_response(question, relevant_chunks):
    """
    Generate a natural language response using OpenAI's chat completion.
    
    Args:
        question (str): The original user question
        relevant_chunks (list): Context chunks retrieved from the database
        
    Returns:
        str: Generated answer combining the context and question
    """
    # Combine context chunks with separator
    context = "\n\n".join(relevant_chunks)
    
    # Create system prompt with instructions and context
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    # Call OpenAI API for completion
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Using GPT-3.5 for cost efficiency
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    return response.choices[0].message


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # Example query and response generation
    question = "tell me about databricks"
    
    # Step 1: Retrieve relevant document chunks
    relevant_chunks = query_documents(question)
    
    # Step 2: Generate natural language response
    answer = generate_response(question, relevant_chunks)
    
    # Display the final answer
    print("\n=== Generated Answer ===")
    print(answer.content)