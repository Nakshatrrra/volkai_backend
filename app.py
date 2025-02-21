from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from unsloth import FastLanguageModel
import torch
import re

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: Optional[float] = 0.8

# Load model globally
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    "nakshatra44/mistral_120k_20feb_v2",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
model = FastLanguageModel.for_inference(model)

def format_prompt(messages: List[Message]) -> str:
    formatted_messages = []
    formatted_messages.append("### Context: ")
    for msg in messages:
        if msg.role == "system":
            formatted_messages.append(f" {msg.content}")
        elif msg.role == "user":
            formatted_messages.append(r" \n\n### Human: " + msg.content)
        elif msg.role == "assistant":
            formatted_messages.append(r" \n\n### Assistant: " + msg.content)

    formatted_messages.append(r" \n\n### Assistant: ")
    return "".join(formatted_messages)


def clean_response(text: str) -> str:
    # Remove <|endoftext|> tokens and any extra whitespace
    cleaned = re.sub(r'<\|endoftext\|>', '', text)
    cleaned = cleaned.strip()
    return cleaned

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        # Format the prompt
        prompt = format_prompt(request.messages)
        print("prompt:", prompt)
        
        # Tokenize and prepare inputs
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
        # Generate output
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=request.max_tokens,
            do_sample=True,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean the response
        clean_text = clean_response(generated_text)
        
        return {"generated_text": clean_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
