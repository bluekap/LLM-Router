from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "router"
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    extra_body: Optional[Dict[str, Any]] = None

    model_config = {
        "extra": "allow",
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
    message: Dict[str, Any]
    finish_reason: Optional[str] = None
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

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]
