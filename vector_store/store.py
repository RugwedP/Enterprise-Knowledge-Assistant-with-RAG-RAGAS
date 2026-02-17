from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import Chroma


def create_vector_store(chunks, persist_directory="chroma_db"):
    """Create or load ChromaDB vector store"""
    
    ## Initialize embeddings (needed for both create and load)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        print(f"Loading cached vector store from {persist_directory}...")

        

        vector_store = Chroma(
            persist_directory=persist_directory,  
            embedding_function=embeddings
        )
       
        return vector_store
    
    # Create new vector store
    print("\nCreating vector store...")
    


   
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory  # Specify where to save
    )
       
    print("Vector store created successfully!\n")
    return vector_store