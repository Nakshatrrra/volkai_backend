from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Iterator, List, Dict
import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread
from fastapi.responses import StreamingResponse

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
FIXED_CONTEXT = "### Context : You are VolkAI, a friendly AI assistant designed for Kairosoft AI Solutions Limited.\n\n"

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
            prompt += f"### Human: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"### Assistant: {msg.content}\n"
    prompt += "### Assistant: "  # Ensure assistant response starts
    return prompt

def generate_response(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> Iterator[str]:
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
    
    if request.stream:
        return StreamingResponse(
            generate_response(
                prompt,
                request.max_tokens,
                request.temperature,
                request.top_p
            ),
            media_type="text/event-stream"
        )
    else:
        response_text = ""
        for text in generate_response(
            prompt,
            request.max_tokens,
            request.temperature,
            request.top_p
        ):
            response_text += text
        return {"response": response_text.strip()}  # Ensure no trailing spaces

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_PATH}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
