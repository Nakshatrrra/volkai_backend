from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Iterator, List, Dict, AsyncGenerator
import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread
from fastapi.responses import StreamingResponse
import json
import asyncio
from contextlib import asynccontextmanager

app = FastAPI(title="Mistral API")

# Model configuration
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Auto-detection
LOAD_IN_4BIT = True

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

MODEL_PATH = "nakshatra44/mistral_120k_20feb_v2"

# Fixed context
FIXED_CONTEXT = "### Context : You are VolkAI, a friendly AI assistant designed for Kairosoft AI Solutions Limited. \n\n" 
# FIXED_CONTEXT = "### Context : \\n\\n" 

# Initialize model and tokenizer at startup
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
model = FastLanguageModel.for_inference(model)
print("Model loaded successfully!")

class Message(BaseModel):
    role: str
    content: str

class GenerationRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 100
    temperature: float = 0.5
    top_p: float = 0.8
    stream: bool = False

def format_prompt(messages: List[Message]) -> str:
    """
    Formats messages in the expected format with the fixed context.
    """
    prompt = FIXED_CONTEXT
    for msg in messages:
        if msg.role == "user":
            prompt += f"### Human: {msg.content}\\n"
        elif msg.role == "assistant":
            prompt += f"### Assistant: {msg.content}\\n"
    prompt += "### Assistant:"  # Ensure assistant response starts
    return prompt

    
async def generate_response(prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Start generation in a separate thread
    thread = Thread(
        target=model.generate,
        kwargs={
            "input_ids": inputs["input_ids"],
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p
        }
    )
    thread.start()

    try:
        yield "data: {\"type\": \"connected\"}\n\n"

        # Process tokens as they come
        for text in streamer:
            if text:
                data = json.dumps({"type": "token", "content": text.replace("\n", " ")})
                yield f"data: {data}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"
    
    except asyncio.CancelledError:
        print("Client disconnected, cleaning up...")
        raise
    finally:
        # Ensure the thread is cleaned up
        if thread.is_alive():
            thread.join(timeout=0.5)

def generate_response_static(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> Iterator[str]:
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    # Setup streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generate in separate thread
    thread = Thread(
        target=model.generate,
        kwargs={
            "input_ids": inputs["input_ids"],
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p
        }
    )
    thread.start()
    
    # Stream the response, stopping at "<|endoftext|>"
    response_text = ""
    for text in streamer:
        if "<|endoftext|>" in text:
            break
        response_text += text
        yield text

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    prompt = format_prompt(request.messages)
    print(f"Prompt: {prompt}")
    
    response_text = ""
    for text in generate_response_static(
        prompt,
        request.max_tokens,
        request.temperature,
        request.top_p
    ):
        response_text += text
    return {"response": response_text.strip()}  # Ensure no trailing spaces


async def stream_tokens(streamer: TextIteratorStreamer) -> AsyncGenerator[str, None]:
    """
    Asynchronously stream tokens as they're generated.
    """
    try:
        # Send initial connection message
        yield "data: {\"type\": \"connected\"}\n\n"
        
        async def async_iterate():
            for text in streamer:
                if text:
                    # Clean and format the token
                    cleaned_text = text.replace("\n", " ")
                    data = json.dumps({"type": "token", "content": cleaned_text})
                    yield f"data: {data}\n\n"
                    # Small delay to allow other tasks to run
                    await asyncio.sleep(0)
        
        async for chunk in async_iterate():
            yield chunk
            
        # Send completion message
        yield "data: {\"type\": \"done\"}\n\n"
    
    except asyncio.CancelledError:
        print("Stream cancelled by client")
        raise
    finally:
        print("Stream completed")

@app.post("/generate_stream")
async def generate_text_stream(request: GenerationRequest):
    prompt = format_prompt(request.messages)
    print(f"Prompt: {prompt}")

    # Prepare input tokens
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Initialize streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Start generation in background thread immediately
    thread = Thread(
        target=model.generate,
        kwargs={
            "input_ids": inputs["input_ids"],
            "streamer": streamer,
            "max_new_tokens": request.max_tokens,
            "do_sample": True,
            "temperature": request.temperature,
            "top_p": request.top_p
        }
    )
    thread.start()

    # Return streaming response immediately
    return StreamingResponse(
        stream_tokens(streamer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
