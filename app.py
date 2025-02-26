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
from pymongo import MongoClient
import re
from contextlib import asynccontextmanager
import queue
from fastapi import HTTPException
from starlette.background import BackgroundTask

app = FastAPI(title="Mistral API")

# MongoDB Connection
MONGO_URI = "mongodb+srv://naksh:naksh@cluster0.3d7ly.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client.volkai_database
collection = db.entity_info

# Model configuration remains the same
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "nakshatra44/mistral_25feb_code_incremental"
FIXED_CONTEXT = "### Context : You are VolkAI, a friendly AI assistant designed for Kairosoft AI Solutions Limited. "

# Initialize model and tokenizer
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
    
def get_entity_info(prompt: str) -> str:
    prompt_lower = prompt.lower()
    
    entities = ["ceo", "md", "coo", "cfo", "cto"]  # Add more entities as needed
    matching_info = []
    
    try:
        for entity in entities:
            if entity in prompt_lower:
                result = collection.find_one({
                    "entity": entity,
                    "company": "Kairosoft"
                })
                print("result: ",result)
                
                if result and 'info' in result:
                    matching_info.append(result['info'])
        print("check1: ",matching_info)
        # Join all matching information with a space
        return " ".join(matching_info) if matching_info else ""
        
    except Exception as e:
        print(f"Error querying MongoDB: {str(e)}")
        return ""

def format_prompt(messages: List[Message]) -> str:
    prompt = FIXED_CONTEXT
    temp_prompt = ""
    for msg in messages:
        if msg.role == "user":
            temp_prompt += f"### Human: {msg.content}"
    additional_context = get_entity_info(temp_prompt)
    if additional_context:
        prompt += f" {additional_context}\n\n"
    else:
        prompt += "\n\n"
    for msg in messages:
        if msg.role == "user":
            prompt += f"### Human: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"### Assistant: {msg.content}\n"
    prompt += "### Assistant:"
    return prompt

async def async_generate(
    inputs: dict,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    queue: asyncio.Queue
):
    try:
        # Create streamer with a small timeout
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=10.0  # Add timeout to prevent blocking
        )
        
        # Start generation in a separate thread
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p
        }
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()
        
        # print("\nGenerated Response:")
        # for text in streamer:
        #     print(text, end="", flush=True)
        
        # Process tokens as they arrive
        async def process_stream():
            for text in streamer:
                if "<|endoftext|>" in text:
                    break
                if text:  # Only process non-empty tokens
                    print(text, end="", flush=True)
                    await queue.put(text)
                    await asyncio.sleep(0)
        
            # Signal completion
            await queue.put(None)
        await process_stream()
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        traceback.print_exc()
        await queue.put(None)

@app.post("/generate_stream")
async def generate_text_stream(request: GenerationRequest):
    prompt = format_prompt(request.messages)
    print(f"Prompt: {prompt}")

    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Create queue for async communication
    token_queue = asyncio.Queue()

    # Start generation in background
    asyncio.create_task(async_generate(
        inputs,
        request.max_tokens,
        request.temperature,
        request.top_p,
        token_queue
    ))

    async def event_generator():
        try:
            yield "data: {\"type\": \"connected\"}\n\n"  
            await asyncio.sleep(0.01)

            while True:
                # Wait for next token with timeout
                try:
                    token = await asyncio.wait_for(token_queue.get(), timeout=10.0)
                    if token is None:  # Generation complete
                        break
                    
                    # Send token
                    data = json.dumps({
                        "type": "token",
                        "content": token
                    })
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.01) 
                    
                except asyncio.TimeoutError:
                    print("Token generation timeout")
                    break

            # Send completion message
            yield "data: {\"type\": \"done\"}\n\n"
            
        except Exception as e:
            print(f"Streaming error: {str(e)}")
            # Send error message to client
            error_data = json.dumps({
                "type": "error",
                "content": "An error occurred during generation"
            })
            yield f"data: {error_data}\n\n"


    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Prevents Nginx buffering if you're using it
        }
    )