from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    extra_body: Optional[Dict[str, Any]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain quantum entanglement in one sentence"
                    }
                ]
            }
        }
    }

class Choice(BaseModel):
    message: Message
    finish_reason: str
    index: int

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class HealthStatus(BaseModel):
    status: str
    active_keys: int
    cooldown_keys: int
    active_models: int
    active_model_list: List[str]
    total_requests_today: int
    provider_health: Dict[str, Any]
