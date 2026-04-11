import time
import asyncio
import logging
from typing import List, Optional, Any, Dict
from litellm import acompletion, exceptions
from .key_manager import KeyManager, Key
from db.models import RequestLog
from db.database import AsyncSessionLocal
from schemas import ChatCompletionRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM-Gateway")

class LLMGateway:
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager

    async def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        retries = 0
        max_retries = min(len(self.key_manager.keys), 5) # Try up to 5 keys or total available
        
        last_exception = None
        
        while retries < max_retries:
            key = await self.key_manager.get_healthiest_key()
            if not key:
                raise Exception("No healthy keys available in the pool.")

            start_time = time.time()
            try:
                logger.info(f"Attempting request with {key.provider} ({key.model_id}) - Try {retries + 1}")
                
                # Use litellm with selected key
                # We ensure the model has the correct provider prefix for LiteLLM
                requested_model = request.model if request.model else key.model_id
                
                # Prefix model with provider if not already present (required for some LiteLLM providers like gemini)
                if key.provider == "gemini" and not requested_model.startswith("gemini/"):
                    model_to_use = f"gemini/{requested_model}"
                elif key.provider == "groq" and not requested_model.startswith("groq/"):
                    model_to_use = f"groq/{requested_model}"
                elif key.provider == "cerebras" and not requested_model.startswith("cerebras/"):
                    model_to_use = f"cerebras/{requested_model}"
                else:
                    model_to_use = requested_model

                # Prepare completion arguments
                completion_args = {
                    "model": model_to_use,
                    "messages": [m.dict() for m in request.messages],
                    "api_key": key.api_key,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": request.stream,
                    **(request.extra_body or {})
                }
                
                # For Gemini AI Studio, LiteLLM sometimes needs GEMINI_API_KEY or GOOGLE_API_KEY.
                # Passing it as api_key with the gemini/ prefix should work, 
                # but we'll be explicit if it helps the driver.
                if key.provider == "gemini":
                    completion_args["gemini_api_key"] = key.api_key

                response = await acompletion(**completion_args)
                
                latency = (time.time() - start_time) * 1000
                
                # Successful request
                await self.key_manager.reset_fail(key)
                await self.key_manager.update_usage(key)
                await self._log_request(key, model_to_use, response, latency, 200)
                
                return response
                
            except (exceptions.RateLimitError, exceptions.ServiceUnavailableError, exceptions.APIError) as e:
                latency = (time.time() - start_time) * 1000
                logger.warning(f"Request failed with {key.provider}: {str(e)}")
                
                # Determine cooldown period
                cooldown_s = 60
                if hasattr(e, 'retry_after') and e.retry_after:
                   try:
                       cooldown_s = int(e.retry_after)
                   except:
                       pass
                
                await self.key_manager.mark_fail(key, cooldown_seconds=cooldown_s)
                await self._log_request(key, key.model_id, None, latency, getattr(e, 'status_code', 500), str(e))
                
                last_exception = e
                retries += 1
                continue # Try next key
                
            except Exception as e:
                logger.error(f"Unexpected error with {key.provider}: {str(e)}")
                last_exception = e
                break # Non-retryable error potentially
                
        if last_exception:
            raise last_exception
        raise Exception("Failed to complete request after multiple retries.")

    async def _log_request(self, key: Key, model: str, response: Optional[Any], latency: float, status_code: int, error: Optional[str] = None):
        async with AsyncSessionLocal() as session:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            if response and hasattr(response, 'usage'):
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                total_tokens = getattr(response.usage, 'total_tokens', 0)
            
            log_entry = RequestLog(
                provider=key.provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency,
                status_code=status_code,
                error_message=error
            )
            session.add(log_entry)
            await session.commit()
