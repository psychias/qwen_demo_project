#!/usr/bin/env python3
"""
Qwen1.5-4B Chat Engine
A simple FastAPI server for serving the Qwen/Qwen1.5-4B model with streaming support.
"""

import asyncio
import json
import logging
import os
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "Qwen/Qwen1.5-0.5B"  # Using Chat variant for better conversational ability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Global model and tokenizer
model = None
tokenizer = None


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = True


def load_model():
    """Download and load the Qwen model and tokenizer."""
    global model, tokenizer
    
    logger.info(f"Loading model {MODEL_ID}...")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Using dtype: {TORCH_DTYPE}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    if DEVICE == "cpu":
        model = model.to(DEVICE)
    
    model.eval()
    logger.info("Model loaded successfully!")
    
    return model, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading model on startup."""
    load_model()
    yield
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Qwen Chat Engine",
    description="API for Qwen1.5-4B Chat model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "device": DEVICE,
        "model_loaded": model is not None
    }


def format_messages_for_qwen(messages: List[Message]) -> str:
    """Format messages into Qwen chat format."""
    # Qwen1.5-Chat uses a specific chat template
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Use the tokenizer's chat template
    text = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


async def generate_stream(request: ChatCompletionRequest):
    """Generate streaming response from the model."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format the prompt
    prompt = format_messages_for_qwen(request.messages)
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    # Create streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    # Generation parameters
    generation_kwargs = {
        **inputs,
        "max_new_tokens": request.max_tokens,
        "temperature": max(request.temperature, 0.01),  # Avoid division by zero
        "top_k": request.top_k,
        "top_p": request.top_p,
        "do_sample": request.temperature > 0,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream tokens
    for token in streamer:
        if token:
            data = json.dumps({"token": token})
            yield f"data: {data}\n\n"
            await asyncio.sleep(0)  # Allow other tasks to run
    
    thread.join()
    yield "data: [DONE]\n\n"


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completion endpoint with streaming support."""
    try:
        return StreamingResponse(
            generate_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "name": "Qwen Chat Engine",
        "model": MODEL_ID,
        "endpoints": {
            "/health": "Health check",
            "/chat/completions": "Chat completion (POST)",
        }
    }


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file."""
    # Look for index.html in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(script_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return {"error": "index.html not found", "looked_in": script_dir}


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Qwen1.5-0.5B Engine")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )