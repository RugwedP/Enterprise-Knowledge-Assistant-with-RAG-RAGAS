from langchain_community.document_loaders import PyPDFLoader
import os

def load_document(folder_path):
    """Load PDFs and merge pages from same document"""
    print("Inside load document")
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found")
        return []
    
    documents = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            
            try:
                # Load all pages from this PDF
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                
                # ✅ MERGE ALL PAGES INTO ONE DOCUMENT
                if pages:
                    merged_content = "\n\n".join([page.page_content for page in pages])
                    merged_doc = pages[0]  # Use first page's metadata
                    merged_doc.page_content = merged_content
                    documents.append(merged_doc)
                    
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    print(f"Loaded {len(documents)} documents")
    return documents