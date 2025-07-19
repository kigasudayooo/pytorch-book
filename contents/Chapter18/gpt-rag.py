from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
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
    Search the vector store for similar documents, removing duplicates.
    
    Args:
        vectorstore (Chroma): The vector store instance
        query (str): The search query
        k (int): Number of results to return
        
    Returns:
        list: List of unique relevant documents
    """
    # Request more results initially to account for potential duplicates
    initial_k = k * 2
    results = vectorstore.similarity_search(query, k=initial_k)
    
    # Use set to track seen content and keep unique documents
    seen_content = set()
    unique_results = []
    
    for doc in results:
        # Create a hash of the content to check for duplicates
        # You could also use other fields like metadata if needed
        content_hash = hash(doc.page_content.strip())
        
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_results.append(doc)
            
            # Break if we have enough unique results
            if len(unique_results) == k:
                break
    
    return unique_results

def query_gpt(prompt, context, model="gpt-4", temperature=0.7):
    """
    Query OpenAI's GPT model with a prompt and context.
    
    Args:
        prompt (str): The user's question
        context (str): Retrieved context from the vector store
        model (str): The model to use (default: gpt-3.5-turbo)
        temperature (float): Generation temperature (default: 0.7)
        
    Returns:
        str: Generated response
    """
    # Initialize the Chat model
    chat = ChatOpenAI(
        model=model,
        temperature=temperature
    )
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following context to answer questions. "
                  "Please provide as much detail as possible in a comprehensive answer."),
        ("system", "Context:\n{context}"),
        ("user", "{question}")
    ])
    
    # Format the prompt with the context and question
    formatted_prompt = prompt_template.format(
        context=context,
        question=prompt
    )
    
    # Get the response
    response = chat.invoke(formatted_prompt)
    
    return response.content

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
    
    # Generate response using GPT
    response = query_gpt(query, context)
    
    return response, relevant_docs

if __name__ == "__main__":
    # Set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Load the existing vector store
    persist_dir = "./chroma_db"  # Change this to your vector store directory
    vectorstore = load_vectorstore(persist_dir)
    
    # Example query
    query = "Give me some details about Soo-Kyung Kim. Where is she from, what does she like?"
    
    # Perform RAG query
    answer, sources = rag_query(vectorstore, query)
    
    # Print results
    print("\nGenerated Answer:")
    print(answer)
    print("\nSources:")
    for i, doc in enumerate(sources, 1):
        print(f"\nSource {i}:")
        print(f"Page: {doc.metadata['page']}")
        print(f"Content: {doc.page_content[:200]}...")

# Requirements:
# pip install langchain langchain-openai langchain-chroma chromadb