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

    @staticmethod
    def _resolve_model_name(provider: str, requested_model: str) -> str:
        """Resolve the model name with the correct LiteLLM provider prefix."""
        # Map specific providers to their required LiteLLM prefix
        provider_prefix_map = {
            "github": "openai",
            "gemini": "gemini",
            "groq": "groq",
            "cerebras": "cerebras"
        }
        
        prefix = provider_prefix_map.get(provider)
        if prefix and not requested_model.startswith(f"{prefix}/"):
            return f"{prefix}/{requested_model}"
            
        return requested_model
    @staticmethod
    def _extract_headers(obj: Any) -> Dict[str, str]:
        """Extract headers from a litellm response or exception.
        
        LiteLLM stores headers in multiple possible locations — we check all of them
        in priority order without early-exit so nothing is missed.
        """
        candidates: Dict[str, str] = {}

        # 1. _hidden_params['additional_headers'] — most reliable for LiteLLM responses
        hidden = getattr(obj, '_hidden_params', None) or {}
        for key in ('additional_headers', 'response_headers'):
            h = hidden.get(key)
            if isinstance(h, dict):
                candidates.update(h)

        # 2. _hidden_params['original_response'].headers
        orig = hidden.get('original_response')
        if orig is not None:
            if hasattr(orig, 'headers') and orig.headers:
                candidates.update(dict(orig.headers))
            elif isinstance(orig, dict) and 'headers' in orig:
                candidates.update(dict(orig['headers']))

        # 3. Direct .headers on the object (exceptions often carry these)
        direct = getattr(obj, 'headers', None)
        if direct:
            candidates.update(dict(direct))

        # 4. obj.response.headers (some exception wrappers)
        resp = getattr(obj, 'response', None)
        if resp is not None:
            resp_headers = getattr(resp, 'headers', None)
            if resp_headers:
                candidates.update(dict(resp_headers))

        result = {str(k).lower(): str(v) for k, v in candidates.items()}
        return result


    @staticmethod
    def _parse_ratelimit_reset(reset_str: Optional[str]) -> int:
        """Parse rate limit reset strings avoiding epochs or strings with 'ms'/'s'."""
        if not reset_str:
            return 60
        try:
            s = str(reset_str).lower().replace('s', '').replace('ms', '').strip()
            val = float(s)
            if val > 1e9:  # Looks like unix epoch
                val = val - time.time()
            return max(int(val), 1)
        except Exception:
            return 60
    async def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        retries = 0
        max_retries = min(len(self.key_manager.keys), 5) # Try up to 5 keys or total available
        
        last_exception = None
        
        # Pre-compute request data once outside the retry loop to improve efficiency
        messages_dict = [m.dict() for m in request.messages]
        extra_body = request.extra_body or {}
        
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
                model_to_use = self._resolve_model_name(key.provider, requested_model)

                # Prepare completion arguments
                completion_args = {
                    "model": model_to_use,
                    "messages": messages_dict,
                    "api_key": key.api_key,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": request.stream,
                    **extra_body
                }
                
                if key.provider == "github":
                    completion_args["api_base"] = "https://models.github.ai/inference"

                # For Gemini AI Studio, LiteLLM sometimes needs GEMINI_API_KEY or GOOGLE_API_KEY.
                # Passing it as api_key with the gemini/ prefix should work, 
                # but we'll be explicit if it helps the driver.
                if key.provider == "gemini":
                    completion_args["gemini_api_key"] = key.api_key

                response = await acompletion(**completion_args)
                
                latency = (time.time() - start_time) * 1000
                
                # Check for rate limits even on successful requests (the 'Header' trick)
                headers = self._extract_headers(response)
                remaining_str = headers.get('x-ratelimit-remaining-requests') or headers.get('x-ratelimit-remaining')
                limit_str = headers.get('x-ratelimit-limit-requests') or headers.get('x-ratelimit-limit')
                
                # Safe int conversion
                def _to_int(v):
                    try: return int(v) if v is not None else None
                    except: return None
                
                remaining = _to_int(remaining_str)
                limit = _to_int(limit_str)
                
                # Persist live header data to DB for dashboard display
                await self.key_manager.update_ratelimit_headers(key, limit, remaining)
                
                if remaining is not None and remaining <= 0:
                    reset = headers.get('x-ratelimit-reset-requests') or headers.get('x-ratelimit-reset')
                    cooldown_s = self._parse_ratelimit_reset(reset)
                    logger.warning(f"Key {key.provider} exhausted. Blacklisting for {cooldown_s}s. (Success)")
                    await self.key_manager.mark_fail(key, cooldown_seconds=cooldown_s)
                else:
                    await self.key_manager.reset_fail(key)
                
                await self.key_manager.update_usage(key)
                await self._log_request(key, model_to_use, response, latency, 200)
                
                return response
                
            except (exceptions.RateLimitError, exceptions.ServiceUnavailableError, exceptions.APIError) as e:
                latency = (time.time() - start_time) * 1000
                logger.warning(f"Request failed with {key.provider}: {str(e)}")
                
                # Extract headers and use them for cooldown
                headers = self._extract_headers(e)
                reset = headers.get('x-ratelimit-reset-requests') or headers.get('x-ratelimit-reset')
                
                cooldown_s = self._parse_ratelimit_reset(reset)
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
