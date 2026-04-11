import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    print("Checking health...")
    response = requests.get(f"{BASE_URL}/health")
    print(json.dumps(response.json(), indent=2))

def test_chat():
    print(f"\nSending chat request (letting gateway auto-select model)...")
    payload = {
        "messages": [
            {"role": "user", "content": "Explain quantum entanglement in one sentence."}
        ]
    }

    
    start = time.time()
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    latency = (time.time() - start) * 1000
    
    if response.status_code == 200:
        print(f"Success! Latency: {latency:.2f}ms")
        print("Response:", response.json()["choices"][0]["message"]["content"])
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    # Note: Make sure the server is running (python main.py)
    # And keys.json is populated with at least one valid key.
    try:
        test_health()
        test_chat()
        test_health() # See if usage count increased
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to gateway. Is it running on port 8000?")
