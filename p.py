import requests

url = "https://dhaara.io/generate_stream"
headers = {"Content-Type": "application/json"}
data = {
    "messages": [{"role": "user", "content": "how to open a bank account?"}],
    "max_tokens": 100,
    "temperature": 0.5,
    "top_p": 0.8,
    "stream": True
}

response = requests.post(url, json=data, stream=True)

for chunk in response.iter_content(chunk_size=1024):
    print(chunk.decode(), end="", flush=True)
