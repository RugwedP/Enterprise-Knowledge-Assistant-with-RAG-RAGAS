from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from rag.loader import load_document
from rag.chunking import chunk_documents
from vector_store.store import create_vector_store
from rag.generator import get_llm
from dotenv import load_dotenv

load_dotenv()

def evaluate_rag():
    # Setup RAG
    docs = load_document("data")
    chunks = chunk_documents(docs)
    vector_store = create_vector_store(chunks, cache_path="vector_store_cache")
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    llm = get_llm()
    
    # Test questions with ground truth answers
    test_data = {
        "question": [
            "What equipment does the company provide for remote work?",
            "How do I request time off?",
            "What are the core working hours?"
        ],
        "ground_truth": [
            "Company provides laptop, monitor (on request for full-time remote), headset, VPN access and security software",
            "Check team calendar, discuss with manager, submit in BambooHR 2 weeks advance, manager approves in 48 hours, block calendar, set Slack status",
            "Core hours are 10:00 AM - 3:00 PM local time with 2 hour response time requirement"
        ]
    }
    
    # Generate answers
    answers = []
    contexts = []
    
    for question in test_data["question"]:
        retrieved_docs = retriever.invoke(question)
        context = [doc.page_content for doc in retrieved_docs]
        contexts.append(context)
        
        full_context = "\n\n".join(context)
        prompt = f"""Answer based on context:

{full_context}

Question: {question}

Answer:"""
        
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        answers.append(answer)
    
    # Create evaluation dataset
    eval_dataset = Dataset.from_dict({
        "question": test_data["question"],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": test_data["ground_truth"]
    })
    
    # Evaluate
    print("\n🔍 Evaluating RAG system with RAGAS...\n")
    result = evaluate(
        eval_dataset,
        metrics=[
            faithfulness(),
            answer_relevancy(),
            context_precision(),
            context_recall()
        ]
    )
    
    print("\n📊 RAGAS Evaluation Results:")
    print("="*60)
    for metric, score in result.items():
        print(f"{metric}: {score:.3f}")
    print("="*60)
    
    return result

if __name__ == "__main__":
    evaluate_rag()