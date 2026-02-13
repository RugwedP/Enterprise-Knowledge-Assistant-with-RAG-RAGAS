from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    print("inside the chunk document ")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )   

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks
