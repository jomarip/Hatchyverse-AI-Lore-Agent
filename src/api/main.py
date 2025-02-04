from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.anthropic import AnthropicEmbeddings
from langchain.embeddings.deepseek import DeepseekEmbeddings
from ..models.lore_validator import LoreValidator
from ..models.chatbot import LoreChatbot
from ..data.data_loader import DataLoader
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hatchyverse Lore Chatbot",
    description="An AI-powered chatbot for exploring and contributing to Hatchyverse lore",
    version="1.0.0"
)

# Initialize components
data_dir = os.getenv("DATA_DIR", "./data")
model_provider = os.getenv("MODEL_PROVIDER", "openai")

# Get provider-specific model names
model_names = {
    "openai": os.getenv("OPENAI_MODEL_NAME", "gpt-4-0125-preview"),
    "anthropic": os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-sonnet-20240229"),
    "deepseek": os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
}

# Get provider-specific embedding models
embedding_models = {
    "openai": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
}

model_name = model_names.get(model_provider)
if not model_name:
    raise ValueError(f"No model name configured for provider: {model_provider}")

vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str
    chat_history: Optional[List[str]] = None

class LoreSubmission(BaseModel):
    content: str
    entity_type: str
    name: str
    element: Optional[str] = None
    metadata: Optional[dict] = None

# Global variables for components
chatbot = None
validator = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    try:
        global chatbot, validator
        
        # Initialize embeddings based on provider
        if model_provider == "openai":
            embeddings = OpenAIEmbeddings(model=embedding_models["openai"])
        elif model_provider == "anthropic":
            embeddings = AnthropicEmbeddings()
        elif model_provider == "deepseek":
            embeddings = DeepseekEmbeddings()
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # Initialize validator
        validator = LoreValidator(embeddings)
        
        # Load existing data
        loader = DataLoader(data_dir)
        entities = loader.load_all_data()
        
        # Build knowledge base
        validator.build_knowledge_base(entities)
        
        # Initialize chatbot with provider
        chatbot = LoreChatbot(
            validator,
            model_provider=model_provider,
            model_name=model_name
        )
        
        logger.info(f"Successfully initialized all components using {model_provider}")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.post("/chat")
async def chat_endpoint(message: ChatMessage):
    """
    Handle chat messages.
    
    Args:
        message: ChatMessage object containing the user's message and optional chat history
        
    Returns:
        Dict containing the chatbot's response
    """
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
            
        response = chatbot.generate_response(
            message.message,
            message.chat_history
        )
        
        return {
            "response": response["response"],
            "validation": response["validation"]
        }
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit")
async def submit_lore(submission: LoreSubmission):
    """
    Handle new lore submissions.
    
    Args:
        submission: LoreSubmission object containing the new lore content
        
    Returns:
        Dict containing validation results and feedback
    """
    try:
        if not validator:
            raise HTTPException(status_code=503, detail="Validator not initialized")
            
        # Format submission for validation
        formatted_submission = f"""
        Name: {submission.name}
        Type: {submission.entity_type}
        Element: {submission.element or 'N/A'}
        Description: {submission.content}
        """
        
        # Check for conflicts
        validation = validator.check_conflict(formatted_submission)
        
        # Process through chatbot for detailed feedback
        response = chatbot.generate_response(
            f"[SUBMIT]{formatted_submission}",
            []
        )
        
        return {
            "response": response["response"],
            "validation": validation,
            "accepted": len(validation["conflicts"]) == 0
        }
        
    except Exception as e:
        logger.error(f"Error in submit endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "chatbot": chatbot is not None,
            "validator": validator is not None
        }
    } 