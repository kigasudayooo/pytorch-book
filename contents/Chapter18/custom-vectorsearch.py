import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import numpy as np

class CustomEmbedding:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_documents(self, texts):
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize and prepare input
                encoded_input = self.tokenizer(
                    text, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                ).to(self.device)
                
                # Generate model output
                model_output = self.model(**encoded_input)
                
                # Pool the outputs into a single embedding
                embedding = self._mean_pooling(
                    model_output,
                    encoded_input['attention_mask']
                )
                
                # Normalize embedding
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                
                # Convert to numpy and store
                embeddings.append(embedding.cpu().numpy()[0])
        
        return embeddings
    
    def embed_query(self, text):
        """Generate embedding for a single query text."""
        return self.embed_documents([text])[0]

class PDFVectorStore:
    def __init__(self, persist_directory="./chroma_db2", model_name='sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize with a specific model. Different models have different embedding dimensions:
        - 'sentence-transformers/all-mpnet-base-v2': 768 dimensions
        - 'sentence-transformers/all-MiniLM-L6-v2': 384 dimensions
        - 'sentence-transformers/all-distilroberta-v1': 768 dimensions
        Choose based on your needs for accuracy vs speed/memory
        """
        self.persist_directory = persist_directory
        self.embedding_model = CustomEmbedding(model_name=model_name)
        self.vectorstore = None
    
    def process_pdf(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        """Process a PDF file and create a vector store."""
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        
        # Persist the vector store
        self.vectorstore.persist()
        
        return self.vectorstore
    
    def search(self, query, k=3):
        """Search the vector store for similar documents."""
        if self.vectorstore is None:
            raise ValueError("Vector store hasn't been created yet. Process a PDF first.")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results

class PDFDataset(Dataset):
    """Dataset for batch processing PDF chunks if needed."""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def main():
    # Example usage
    pdf_path = "space-cadets-2020-master.pdf"
    
    # Initialize vector store
    vector_store = PDFVectorStore()
    
    # Process PDF and create vector store
    vectorstore = vector_store.process_pdf(pdf_path)
    
    # Example search
    query = "Tell me about Soo-Kyung Kim. Where is she from? What does she like?"
    results = vector_store.search(query)
    
    # Print results
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {doc.page_content}")
        print(f"Source: Page {doc.metadata['page']}")
        print(f"Start Index: {doc.metadata.get('start_index', 'N/A')}")

if __name__ == "__main__":
    main()

# Requirements:
# pip install torch transformers langchain langchain-community chromadb pypdf sentence-transformers