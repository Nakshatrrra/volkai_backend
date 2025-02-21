from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch

# Define FastAPI app
app = FastAPI()

# Define request model
class RequestBody(BaseModel):
    prompt: str
    max_tokens: int = 100

# Load the fine-tuned model and tokenizer
max_seq_length = 2048
dtype = None
load_in_4bit = True

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    "nakshatra44/mistral_120k_20feb_v2",  # Path to your fine-tuned model
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.for_inference(model)
print("Model loaded successfully.")

@app.post("/generate")
def generate_response(request: RequestBody):
    try:
        # Tokenize input prompt
        inputs = tokenizer(request.prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        # Generate response
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.8,
        )

        # Decode generated tokens
        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        return {"response": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI server (use `uvicorn filename:app --reload` to start)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)