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
from sumy.summarizers.lsa import LsaSummarizer
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
    """Scrapes a webpage using the Taras Scraper API."""
    print(f"🔍 Scraping: {url}")
    try:
        response = requests.get(SCRAPER_API, params={'url': url})
        response.raise_for_status()
        # Extract clean text from HTML
        content = response.text 
        cleaned_text = clean_text(content)
        return cleaned_text
    except requests.RequestException as e:
        print(f"⚠️ Scraping failed for {url}: {e}")
        return None


@app.get("/search_summary")
def search_summary(query: str):
    """Fetches Google search results, scrapes the first successful page, and summarizes content."""
    try:
        # Google Custom Search API request
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": GOOGLE_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "num": 2  # Fetch top 2 results
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "items" not in data:
            return {"message": "No search results found"}

        # Extract top 2 results
        top_results = [item.get("link") for item in data["items"][:2]]
        print(f"🔎 Top search results: {top_results}")

        # Try scraping first URL
        content = scrape_page(top_results[0])
        if not content and len(top_results) > 1:
            print(f"❌ Scraping failed for {top_results[0]}. Trying second URL...")
            content = scrape_page(top_results[1])

        # If scraping failed for both, return links instead
        if not content:
            print("❌ Scraping failed for both URLs. Returning links instead.")
            return {
                "query": query,
                "message": "Scraping failed. Here are the links:",
                "sources": top_results
            }

        # Summarize the scraped content
        # summary = summarize_content(content)

        # Return the summary if successful
        if content:
            return {
                "query": query,
                "content": content,
                "source": top_results[0] if content else top_results[1]
            }
        else:
            return {
                "query": query,
                "message": "Summarization failed. Here are the links:",
                "sources": top_results
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))