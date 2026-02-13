from langchain_community.document_loaders import PyPDFLoader
import os


def load_document(folder_path):
    print("Inside load document")
    documents = []


    if not os.path.exists(folder_path):
        print(f"Error: Folder path '{folder_path}' does not exist")
        return documents
    
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())
    return documents


