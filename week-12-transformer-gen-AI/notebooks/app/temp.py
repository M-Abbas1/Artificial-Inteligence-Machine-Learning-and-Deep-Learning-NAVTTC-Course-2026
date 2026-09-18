import requests

# URL = "https://abc123.ngrok-free.app/generate"
URL = "https://cavalry-rasping-unvalued.ngrok-free.dev/generate"

data = {
    "prompt": "Artificial intelligence is",
    "max_new_tokens": 50
}

response = requests.post(URL, json=data)

print("Status:", response.status_code)
print("Response:")
print(response.json())