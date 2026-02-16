from rag.loader import load_document
from rag.chunking import chunk_documents
from vector_store.store import create_vector_store
from rag.generator import get_llm
from dotenv import load_dotenv

load_dotenv()

def main():

    docs = load_document("data")
    if not docs:
        print("No documents found in 'data' folder!")
        return
    print(f"Loaded {len(docs)} documents")


    chunks = chunk_documents(docs)

    if not chunks:
        print("No chunks created!")
        return
    print(f"Created {len(chunks)} chunks")

    print("\nCreating vector store...")
    

    try:

        vector_store = create_vector_store(chunks, cache_path="vector_store_cache")
        retriever = vector_store.as_retriever(search_kwargs={"k": 10}) 
        print("Vector store created")
    except Exception as e:
        print(e)
        return
    

    try:
        llm = get_llm()
        print("LLM initialized")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return
    
    queries = [
        "What are the core working hours for remote employees?",
        "What equipment does the company provide for remote work?",
        "How do I request time off?",
        "Give contact information of hr",
        "what are the Security and Confidentiality when i will be working remotely?"
        
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}\n")
        
        try:
            retrieved_docs = retriever.invoke(query)
            
           
            
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
        
            
            # Create prompt
            prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't have that information."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
            
            # Get response from LLM
            response = llm.invoke(prompt)
            
            # Handle different response types
            if hasattr(response, 'content'):
                answer = response.content
            elif isinstance(response, str):
                answer = response
            else:
                answer = str(response)
            
            print(f"\nANSWER:\n{answer}")
            
        except Exception as e:
            print(f"Error processing query: {e}")
    


if __name__ == "__main__":
    main()

