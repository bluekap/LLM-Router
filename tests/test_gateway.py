import asyncio
import sys
import os

# Add the parent directory to sys.path so we can import from main project files
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import ChatCompletionRequest, Message
from core.gateway import LLMGateway
from core.key_manager import KeyManager

async def main():
    km = KeyManager()
    km.keys.append(type('obj', (object,), {'provider': 'gemini', 'model_id': 'gemini-1.5-flash-preview', 'api_key': 'dummy', 'key_hash': 'h', 'priority': 1, 'daily_limit': None})())
    
    gateway = LLMGateway(km)
    req = ChatCompletionRequest(
        model="gemini-1.5-flash-preview",
        messages=[Message(role="user", content="hello")],
        stream=False
    )
    
    try:
        resp = await gateway.chat_completion(req)
        print("Response:", resp)
    except Exception as e:
        print("Error:", type(e), str(e))

asyncio.run(main())
