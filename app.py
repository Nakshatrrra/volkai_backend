from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch

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

# ✅ Load the model & tokenizer
model_repo = "nakshatra44/mistral_120k_20feb_v2"
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_repo,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

model = FastLanguageModel.for_inference(model)

# ✅ Define request format
class Request(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.5
    top_p: float = 0.8


# ✅ Function to clean up the response
def clean_response(response_text):
    if "### Assistant:" in response_text:
        parts = response_text.split("### Assistant:", 1)
        response_text = parts[1].strip()

    response_text = response_text.split("<|endoftext|>", 1)[0].strip()
    return response_text

# ✅ API Endpoint
@app.post("/generate")
def generate_text(request: Request):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    output_tokens = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    cleaned_text = clean_response(output_text)

    return {"response": cleaned_text}
