from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import requests
import os

def load_vectorstore(persist_directory="./chroma_db"):
    """
    Load an existing vector store from a directory.
    
    Args:
        persist_directory (str): Directory where the vector store is persisted
        
    Returns:
        Chroma: The vector store instance
    """
    embeddings = OpenAIEmbeddings()
    
    # Load existing vector store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vectorstore

def search_vectorstore(vectorstore, query, k=3):
    """
    Search the vector store for similar documents.
    
    Args:
        vectorstore (Chroma): The vector store instance
        query (str): The search query
        k (int): Number of results to return
        
    Returns:
        list: List of relevant documents
    """
    results = vectorstore.similarity_search(query, k=k)
    return results

def query_ollama(prompt, context, model="llama3.1:latest", temperature=0.7):
    """
    Query the Ollama server with a prompt and context.
    
    Args:
        prompt (str): The user's question
        context (str): Retrieved context from the vector store
        model (str): The model to use (default: gemma2:2b)
        temperature (float): Generation temperature (default: 0.7)
        
    Returns:
        str: Generated response
    """
    ollama_url = "http://localhost:11434/api/chat"
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Use the provided context to answer questions. If you cannot find the answer in the context, say so. Only use information from the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {prompt}"
        }
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature
    }
    
    try:
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error querying Ollama: {str(e)}"

def rag_query(vectorstore, query, num_contexts=3):
    """
    Perform a RAG query using the vector store and Ollama.
    
    Args:
        vectorstore (Chroma): The vector store instance
        query (str): User's question
        num_contexts (int): Number of context passages to retrieve
        
    Returns:
        tuple: (Generated answer, List of source documents)
    """
    # Retrieve relevant documents
    relevant_docs = search_vectorstore(vectorstore, query, k=num_contexts)
    
    # Combine context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Generate response using Ollama
    response = query_ollama(query, context)
    
    return response, relevant_docs

if __name__ == "__main__":
    # Set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Load the existing vector store
    persist_dir = "./chroma_db"  # Change this to your vector store directory
    vectorstore = load_vectorstore(persist_dir)
    
    # Example query
    query = "Please tell me all about Soo-Kyung Kim."
    
    # Perform RAG query
    answer, sources = rag_query(vectorstore, query, num_contexts=10)
    
    # Print results
    print("\nGenerated Answer:")
    print(answer)
    print("\nSources:")
    for i, doc in enumerate(sources, 1):
        print(f"\nSource {i}:")
        print(f"Page: {doc.metadata['page']}")
        print(f"Content: {doc.page_content[:200]}...")

# Requirements:
# pip install langchain langchain-community chromadb openai requests