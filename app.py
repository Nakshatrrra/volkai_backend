from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch
from transformers import TextIteratorStreamer
from threading import Thread
from typing import Optional
import asyncio

app = FastAPI()

# Model configuration
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Auto-detection
LOAD_IN_4BIT = True

# Load model and tokenizer globally
model, tokenizer = FastLanguageModel.from_pretrained(
    "nakshatra44/mistral_120k_20feb_v2",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
model = FastLanguageModel.for_inference(model)

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 0.8

async def generate_text_stream(prompt: str, max_tokens: int, temperature: float, top_p: float):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    # Set up streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generate in separate thread
    thread = Thread(
        target=model.generate,
        kwargs={
            "input_ids": inputs["input_ids"],
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p
        }
    )
    thread.start()
    
    # Stream the response
    for text in streamer:
        yield text
        # Small sleep to prevent overwhelming the client
        await asyncio.sleep(0.01)

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    try:
        return StreamingResponse(
            generate_text_stream(
                request.prompt,
                request.max_tokens,
                request.temperature,
                request.top_p
            ),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)