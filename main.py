from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from openai import OpenAI
import json
import os
import asyncio
import httpx
from loguru import logger
from config import settings
import uvicorn
from mangum import Mangum
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logger.add("logs/faq_bot.log", rotation="500 MB")

app = FastAPI(
    title="FAQ Bot API",
    description="A FastAPI-based FAQ bot that uses OpenAI embeddings for question matching",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Global variable to store the server URL for self-pinging
server_url = None

class FAQItem(BaseModel):
    question: str
    answer: str

class FAQList(BaseModel):
    faqs: List[FAQItem]

class Question(BaseModel):
    text: str

def ensure_data_directory():
    """Ensure the data directory exists"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using OpenAI API"""
    try:
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating embedding")

def save_embeddings(questions: List[str], embeddings: np.ndarray, path: str):
    """Save embeddings to a file"""
    try:
        np.save(path, embeddings)
        logger.info(f"Saved embeddings to {path}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving embeddings")

def load_embeddings(path: str) -> np.ndarray:
    """Load embeddings from a file"""
    try:
        return np.load(path)
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading embeddings")

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global server_url
    ensure_data_directory()
    
    # Set server URL for self-pinging (for Render deployment)
    if os.getenv("RENDER_EXTERNAL_URL"):
        server_url = os.getenv("RENDER_EXTERNAL_URL")
    elif os.getenv("PORT"):
        # For local development or other deployments
        server_url = f"http://localhost:{os.getenv('PORT', '8001')}"
    else:
        server_url = "http://localhost:8001"
    
    logger.info(f"Server URL set to: {server_url}")
    
    # Start background ping task if enabled
    if settings.ENABLE_AUTO_PING:
        asyncio.create_task(background_ping_task())
        logger.info("Background ping task started")
    
    logger.info("Application started")

@app.post("/faqs", response_model=dict)
async def add_faqs(faq_list: FAQList):
    """
    Add new FAQs to the system and generate embeddings
    
    - **faqs**: List of FAQ items containing questions and answers
    """
    try:
        # Load existing FAQs if file exists
        existing_faqs = []
        if os.path.exists(settings.FAQ_FILE):
            with open(settings.FAQ_FILE, 'r') as f:
                existing_faqs = json.load(f)
        
        # Append new FAQs
        new_faqs = [faq.dict() for faq in faq_list.faqs]
        updated_faqs = existing_faqs + new_faqs
        
        # Save updated FAQs to JSON file
        with open(settings.FAQ_FILE, 'w') as f:
            json.dump(updated_faqs, f)
        
        # Load existing embeddings if file exists
        existing_embeddings = np.array([])
        if os.path.exists(settings.EMBEDDINGS_FILE):
            existing_embeddings = load_embeddings(settings.EMBEDDINGS_FILE)
        
        # Generate embeddings for new questions
        new_questions = [faq.question for faq in faq_list.faqs]
        new_embeddings = [get_embedding(q) for q in new_questions]
        new_vectors_np = np.array(new_embeddings).astype("float32")
        
        # Combine existing and new embeddings
        if len(existing_embeddings) > 0:
            updated_embeddings = np.vstack((existing_embeddings, new_vectors_np))
        else:
            updated_embeddings = new_vectors_np
        
        # Save updated embeddings
        save_embeddings(new_questions, updated_embeddings, settings.EMBEDDINGS_FILE)
        
        logger.info(f"Added {len(faq_list.faqs)} new FAQs")
        return {
            "message": f"Successfully added {len(faq_list.faqs)} FAQs",
            "total_faqs": len(updated_faqs)
        }
    except Exception as e:
        logger.error(f"Error adding FAQs: {str(e)}")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=dict)
async def ask_question(question: Question):
    """
    Ask a question and get the most relevant answer
    
    - **text**: The question text to find an answer for
    """
    try:
        # Load FAQs and embeddings
        with open(settings.FAQ_FILE, 'r') as f:
            faqs = json.load(f)
        
        embeddings = load_embeddings(settings.EMBEDDINGS_FILE)
        
        # Get embedding for the question
        question_embedding = get_embedding(question.text)
        
        # Calculate similarities
        similarities = [cosine_similarity(question_embedding, emb) for emb in embeddings]
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
        
        if max_similarity < settings.SIMILARITY_THRESHOLD:
            logger.info(f"No matching FAQ found for question: {question.text}")
            return {
                "answer": "I'm sorry, I couldn't find a relevant answer to your question.",
                "confidence": float(max_similarity)
            }
        
        logger.info(f"Found matching FAQ with confidence: {max_similarity}")
        return {
            "answer": faqs[max_similarity_idx]["answer"],
            "confidence": float(max_similarity)
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def ping_server():
    """Ping the server to keep it alive"""
    global server_url
    if not server_url:
        logger.warning("Server URL not set, cannot ping server")
        return
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server_url}/health")
            if response.status_code == 200:
                logger.info("Server pinged successfully")
            else:
                logger.warning(f"Server ping failed with status: {response.status_code}")
    except Exception as e:
        logger.error(f"Error pinging server: {str(e)}")

async def background_ping_task():
    """Background task that periodically pings the server"""
    while True:
        try:
            await asyncio.sleep(settings.PING_INTERVAL)
            await ping_server()
        except Exception as e:
            logger.error(f"Error in background ping task: {str(e)}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server is running"""
    return {
        "status": "healthy",
        "message": "FAQ Bot API is running",
        "timestamp": asyncio.get_event_loop().time()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True) 


handler = Mangum(app)