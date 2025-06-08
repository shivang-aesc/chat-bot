from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    SIMILARITY_THRESHOLD: float = 0.85
    EMBEDDINGS_FILE: str = "data/faq_embeddings.npy"
    FAQ_FILE: str = "data/faqs.json"

settings = Settings() 