from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Iterator
import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread
from fastapi.responses import StreamingResponse
import json

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

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.5
    top_p: float = 0.8
    stream: bool = False

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
    
    # Stream the response
    for text in streamer:
        yield text

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    if request.stream:
        # Return streaming response
        return StreamingResponse(
            generate_response(
                request.prompt,
                request.max_new_tokens,
                request.temperature,
                request.top_p
            ),
            media_type="text/event-stream"
        )
    else:
        # Return complete response
        response_text = ""
        for text in generate_response(
            request.prompt,
            request.max_new_tokens,
            request.temperature,
            request.top_p
        ):
            response_text += text
        return {"response": response_text}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_PATH}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)