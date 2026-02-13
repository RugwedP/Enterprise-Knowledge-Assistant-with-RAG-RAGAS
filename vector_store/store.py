from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

def create_vector_store(chunks):
    """Create FAISS vector store from document chunks"""
    embeddings = OllamaEmbeddings(model="llama3.2",show_progress=True)
    
    # Test with a single chunk first
    print("Testing embedding generation...")
    try:
        test_embedding = embeddings.embed_query("test")
        print(f"✓ Embedding generated successfully (dimension: {len(test_embedding)})")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        print("Make sure Ollama is running: ollama serve")
        raise
    
    # Create FAISS vector store
    print("Creating vector store...")
    vector_store = Chroma.from_documents(chunks, embeddings)
    
    print(f"✓ Vector store created successfully!")
    return vector_store