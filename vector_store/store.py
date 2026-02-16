from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import Chroma


def create_vector_store(chunks, cache_path="vector_store_cache"):
    """Create or load cached vector store"""
    
    # Check if cache exists
    if os.path.exists(cache_path):
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        vector_store = Chroma(
            cache_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
       
        return vector_store
    
    # Create new vector store
    print("\nCreating vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  # Better embeddings
    )


   
    vector_store = Chroma.from_documents(chunks, embeddings)
    
   
    vector_store.persist()
    
    print("Vector store created successfully!\n")
    return vector_store