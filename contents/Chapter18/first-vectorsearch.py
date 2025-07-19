from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

def create_vectorstore(pdf_path, persist_directory="./chroma_db"):
    """
    Create a vector store from a PDF file using LangChain and Chroma.
    
    Args:
        pdf_path (str): Path to the PDF file
        persist_directory (str): Directory to persist the vector store
        
    Returns:
        Chroma: The vector store instance
    """
    # Load the PDF file
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    texts = text_splitter.split_documents(documents)
    
    # Initialize OpenAI embeddings
    # Make sure to set your OPENAI_API_KEY environment variable
    embeddings = OpenAIEmbeddings()
    
    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the vector store
    vectorstore.persist()
    
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

# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Path to your PDF file
    pdf_path = "space-cadets-2020-master.pdf"
    
    # Create the vector store
    vectorstore = create_vectorstore(pdf_path)
    
    # Example search
    query = "Give me some details about the Soo-Kyung Kim. Where is she from, what does she like, tell me all about her?"
    results = search_vectorstore(vectorstore, query, 5)
    
    # Print results
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {doc.page_content}")
        print(f"Source: Page {doc.metadata['page']}")
        print(f"Start Index: {doc.metadata.get('start_index', 'N/A')}")

# Requirements:
# pip install langchain langchain-community chromadb openai pypdf