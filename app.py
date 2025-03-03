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
import requests
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from transformers import pipeline

# Load environment variables
load_dotenv()

# Get API keys from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
# Web scraping API endpoint
SCRAPER_API = "https://taras-scrape2md.web.val.run/"


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

MODEL_PATH = "nakshatra44/mistral_120k_20feb_v2"
MODEL_PATH_internet = "nakshatra44/only_context_qna"
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

# Initialize model and tokenizer
print("internet Loading model...")
model_internet, tokenizer_internet = FastLanguageModel.from_pretrained(
    MODEL_PATH_internet,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
model_internet = FastLanguageModel.for_inference(model_internet)
print("internet Model loaded successfully!")

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
    # prev_context = ""
    # for msg in messages:
    #     if msg.role == "system":
    #         prev_context += f"{msg.content}"
    additional_context = get_entity_info(temp_prompt)
    # if prev_context:
    #     prompt += f" {prev_context}"
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


def format_prompt_without(messages: List[Message]) -> str:
    prompt = "### Context: "
    for msg in messages:
        if msg.role == "system":
            prompt += f"{msg.content}"
    temp_prompt = ""
    for msg in messages:
        if msg.role == "user":
            temp_prompt += f"### Human: {msg.content}"
    # prev_context = ""
    # for msg in messages:
    #     if msg.role == "system":
    #         prev_context += f"{msg.content}"
    additional_context = get_entity_info(temp_prompt)
    # if prev_context:
    #     prompt += f" {prev_context}"
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

async def async_generate_context(
    inputs: dict,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    queue: asyncio.Queue
):
    try:
        # Create streamer with a small timeout
        streamer = TextIteratorStreamer(
            tokenizer_internet,
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
        
        thread = Thread(target=model_internet.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()
        
        # print("\nGenerated Response:")
        # for text in streamer:
        #     print(text, end="", flush=True)
        
        # Process tokens as they arrive
        async def process_stream():
            for text in streamer:
                if "<|endoftext|>" in text:
                    text = text.replace("<|endoftext|>", "")
                    await queue.put(text)
                    await asyncio.sleep(0)
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
                    text = text.replace("<|endoftext|>", "")
                    await queue.put(text)
                    await asyncio.sleep(0)
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
    
@app.post("/generate_stream_withoutcontext")
async def generate_text_stream(request: GenerationRequest):
    prompt = format_prompt_without(request.messages)
    print(f"Prompt: {prompt}")

    # Prepare inputs
    inputs = tokenizer_internet(prompt, return_tensors="pt")
    inputs = {key: value.to(model_internet.device) for key, value in inputs.items()}

    # Create queue for async communication
    token_queue = asyncio.Queue()

    # Start generation in background
    asyncio.create_task(async_generate_context(
        inputs,
        request.max_tokens,
        0.7,
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
    
@app.get("/google_search")
def google_search(query: str):
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": GOOGLE_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "num": 1  # Number of results
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "items" not in data:
            return {"message": "No results found"}

        results = []
        for item in data["items"]:
            snippet = item.get("snippet", "")
            pagemap = item.get("pagemap", {})

            # Extract richer descriptions if available
            if "metatags" in pagemap and len(pagemap["metatags"]) > 0:
                meta_desc = pagemap["metatags"][0].get("og:description")  # Open Graph Description
                if meta_desc:
                    snippet += f"\n{meta_desc}"  # Append extra description

            results.append({
                "title": item.get("title"),
                "snippet": snippet,
                "link": item.get("link")
            })

        return {"query": query, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@app.get("/duckduckgo_search")
def duckduckgo_search(query: str):
    try:
        response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
        data = response.json()

        # Extract the best available information
        abstract = data.get("Abstract", "").strip()
        results = data.get("Results", [])
        related_topics = data.get("RelatedTopics", [])

        # If abstract is empty, use the first related topic
        if not abstract and related_topics:
            first_related = related_topics[0]
            abstract = first_related.get("Text", "")
            first_url = first_related.get("FirstURL", "")
        else:
            first_url = ""

        return {
            "query": query,
            "abstract": abstract,
            "url": first_url,  # Provide the related topic URL if abstract is empty
            "results": [{"title": r["Text"], "url": r["FirstURL"]} for r in results],
            "related_topics": [{"title": t["Text"], "url": t["FirstURL"]} for t in related_topics if "Text" in t]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def clean_text(html_content):
    # Step 1: Convert HTML to plain text
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")  # Extract text with line breaks

    # Step 2: Remove markdown-style image links
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Step 3: Remove standard markdown links [text](URL)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'\1', text)

    # Step 4: Handle the specific link format from the example
    text = re.sub(r'\[(.*?)\]\((https?://.*?\.val\.town/.*?)\s*".*?"\)', r'\1', text)

    # Step 5: Remove standalone URLs
    text = re.sub(r'https?://\S+', '', text)

    # Step 6: Remove citation references like [1], [2], [a], etc.
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[[a-z]\]', '', text)

    # Step 7: Remove citation note references like [\\[1\\]]
    text = re.sub(r'\[\\+\[\d+\\+\]\]', '', text)

    # Step 8: Clean up wiki formatting for IPA and other special notations
    text = re.sub(r'\[\\\[(.*?)\\\]\]', r'\1', text)  

    # Step 9: Remove empty bullet points and list markers
    text = re.sub(r'^\s*\*\s*$', '', text, flags=re.MULTILINE)

    # Step 10: Remove extra whitespace and newlines
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()

    return text

def scrape_page(url):
    """Scrapes a webpage and extracts 10 meaningful lines of content."""
    print(f"🔍 Scraping: {url}")
    
    try:
        response = requests.get(SCRAPER_API, params={'url': url}, timeout=15)
        response.raise_for_status()
        
        # Extract and clean text
        content = response.text
        cleaned_text = clean_text(content)
        
        # Split into lines and filter out empty lines
        lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
        
        # Extract 10 meaningful lines (with reasonable length)
        meaningful_lines = []
        for line in lines:
            if len(line) > 20 and not line.startswith('Skip to') and not line.startswith('Copyright'):
                meaningful_lines.append(line)
                if len(meaningful_lines) == 10:
                    break
        
        extracted_content = '\n'.join(meaningful_lines)
        print(f"📜 Extracted Content from {url}:", extracted_content[:100] + "...")  # Print preview
        return extracted_content
    
    except requests.Timeout:
        print(f"⏱️ Timeout error while scraping {url}")
        return None
    except requests.ConnectionError:
        print(f"🔌 Connection error while scraping {url}")
        return None
    except requests.RequestException as e:
        print(f"⚠️ Scraping failed for {url}: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error while scraping {url}: {type(e).__name__}: {e}")
        return None

# Load your fine-tuned model
MODEL_PATH_SUMMARIZER = "nakshatra44/mistral_summarizer_1"


# Load model and tokenizer
print("Loading summarization model...")
model_summarizer, tokenizer_summarizer = FastLanguageModel.from_pretrained(
    MODEL_PATH_SUMMARIZER,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
model_summarizer = FastLanguageModel.for_inference(model_summarizer)
print("Model loaded successfully!")

def summarize_with_mistral(text):
    """Uses the fine-tuned Mistral model to generate a summary."""

    # Limit input text to first 10,000 tokens
    tokens = tokenizer_summarizer(text, return_tensors="pt", truncation=True, max_length=10000)
    truncated_text = tokenizer_summarizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

    prompt = f"""### Instruction:
Summarize the following text in a concise manner.

### Text:
{truncated_text}

### Summary:
"""

    # Tokenize input with truncation
    inputs = tokenizer_summarizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)

    # Generate output
    with torch.no_grad():
        output_tokens = model_summarizer.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            max_new_tokens=200,  # Ensures proper summary length
            do_sample=False
        )

    # Decode and clean the output
    summary = tokenizer_summarizer.decode(output_tokens[0], skip_special_tokens=True)
    # Remove unwanted parts like "### Human:" or any additional formatting
    summary = summary.split("### Summary:")[-1].strip()
    summary = summary.split("###")[0].strip()
    return summary.split("### Summary:")[-1].strip()


@app.get("/search_summary")
def search_summary(query: str):
    """Fetches Google search results, scrapes the top 3 pages, and summarizes content."""
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": GOOGLE_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "num": 3  # Request more results in case some scraping fails
        }

        # Google API error handling
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.Timeout:
            return {"error": "Search API timeout", "message": "The search request timed out."}
        except requests.ConnectionError:
            return {"error": "Network error", "message": "Could not connect to search API."}
        except requests.HTTPError as e:
            status_code = e.response.status_code
            if status_code == 403:
                return {"error": "API quota exceeded or credentials invalid", "code": status_code}
            elif status_code == 429:
                return {"error": "Rate limit exceeded", "code": status_code}
            else:
                return {"error": f"HTTP error {status_code}", "message": str(e)}
        except Exception as e:
            return {"error": "Search API error", "message": str(e)}

        if "items" not in data or not data["items"]:
            return {"message": "No search results found for query", "query": query}

        top_results = [item.get("link") for item in data["items"][:5]]
        print(f"🔎 Top search results: {top_results}")

        # Track successful scrapes and content
        successful_scrapes = []
        all_content_pieces = []
        sources = []

        # Try to scrape at least 3 sites successfully
        for url in top_results:
            if len(successful_scrapes) > 3:
                break
                
            content = scrape_page(url)
            if content:
                successful_scrapes.append(url)
                all_content_pieces.append(content)
                sources.append(url)
                print(f"✅ Successfully scraped content from {url}")
            else:
                print(f"❌ Failed to scrape content from {url}")

        # Check if we have enough successful scrapes
        if not successful_scrapes:
            return {
                "query": query,
                "error": "Scraping failed",
                "message": "Could not successfully scrape any of the search results.",
                "attempted_sources": top_results[:5]
            }

        # Combine content from all successful scrapes with source attribution
        combined_content = ""
        for i, content in enumerate(all_content_pieces):
            combined_content += f"SOURCE {i+1} ({successful_scrapes[i]}):\n{content}\n\n"

        # Generate summary of the combined content
        try:
            summary = summarize_with_mistral(combined_content)
        except Exception as e:
            print(f"⚠️ Summarization failed: {e}")
            # Fallback to a simpler method if advanced summarization fails
            summary = combined_content[:500] + "...\n[Summarization failed, showing truncated content]"

        return {
            "query": query,
            "summary": summary,
            "sources": sources,
            "content_preview": combined_content[:300] + "..." if len(combined_content) > 300 else combined_content
        }

    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e)
        print(f"❌ Unhandled exception: {error_type}: {error_details}")
        traceback_info = traceback.format_exc()
        
        # Log the full error but return a cleaner message to the user
        print(f"Traceback: {traceback_info}")
        
        return {
            "error": "Internal server error",
            "error_type": error_type,
            "message": "An unexpected error occurred while processing your request.",
            "query": query
        }