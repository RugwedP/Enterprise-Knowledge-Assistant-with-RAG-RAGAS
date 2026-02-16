from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """Initialize Groq LLM with error handling"""
    
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables. "
            "Please add it to your .env file."
        )
    
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",  # Best model
            groq_api_key=api_key,
            temperature=0,  # Deterministic responses
            max_tokens=1024,  # Response length limit
            timeout=30,  # 30 second timeout
            max_retries=2  # Retry on failure
        )
        return llm
    except Exception as e:
        print(f"Error initializing Groq: {e}")
        raise