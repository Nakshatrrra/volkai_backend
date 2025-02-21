from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# ✅ Load the fine-tuned Mistral model
model_repo = "nakshatra44/mistral_120k_20feb_v2"
max_seq_length = 2048
load_in_4bit = True

tokenizer = AutoTokenizer.from_pretrained(model_repo)
model, _ = FastLanguageModel.from_pretrained(
    model_repo,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.for_inference(model)

# ✅ Define request format
class Request(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.95

# ✅ Streaming Generator Function
def generate_stream(prompt, max_tokens, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    thread = Thread(target=model.generate, kwargs={
        "input_ids": inputs["input_ids"],
        "streamer": streamer,
        "max_new_tokens": max_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
    })
    thread.start()
    
    for text in streamer:
        yield text

# ✅ API Endpoint for Streaming Text Generation
@app.post("/generate")
def generate_text(request: Request):
    return StreamingResponse(generate_stream(request.prompt, request.max_tokens, request.temperature, request.top_p), media_type="text/plain")