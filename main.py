import logging
import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from core.key_manager import KeyManager
from core.gateway import LLMGateway
from schemas import ChatCompletionRequest, ChatCompletionResponse, HealthStatus
from db.database import engine, Base, AsyncSessionLocal
from db.models import KeyMetadata, RequestLog
from sqlalchemy import select, func

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM-Gateway")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized.")
    
    # Sync keys
    await key_manager.sync_with_db()
    
    yield

app = FastAPI(title="Unified LLM Gateway", version="1.0.0", lifespan=lifespan)

# Initialize components
key_manager = KeyManager()
gateway = LLMGateway(key_manager)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("static/dashboard.html", "r") as f:
        return f.read()

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    try:
        response = await gateway.chat_completion(request)
        return response
    except Exception as e:
        logger.error(f"Gateway error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthStatus)
async def health():
    async with AsyncSessionLocal() as session:
        # Get key stats
        stmt = select(KeyMetadata)
        result = await session.execute(stmt)
        all_keys = result.scalars().all()
        
        active_keys = [k for k in all_keys if k.status == "active"]
        cooldown_keys = [k for k in all_keys if k.status == "cooldown"]
        
        # Get total requests today
        today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        stmt_logs = select(func.count(RequestLog.id)).where(RequestLog.timestamp >= today_start)
        count_result = await session.execute(stmt_logs)
        total_today = count_result.scalar()
        
        # Build a hash -> Key map so we can look up daily_limit from in-memory config
        key_limit_map = {k.key_hash: k.daily_limit for k in key_manager.keys}

        provider_health = {}
        for k in all_keys:
            if k.provider not in provider_health:
                provider_health[k.provider] = {"active": 0, "cooldown": 0, "fail_count": 0, "keys": []}
            
            if k.status == "active":
                provider_health[k.provider]["active"] += 1
            else:
                provider_health[k.provider]["cooldown"] += 1
            provider_health[k.provider]["fail_count"] += k.fail_count
            
            provider_health[k.provider]["keys"].append({
                "hash": k.api_key_hash[:8] if k.api_key_hash else "unknown",
                "model_id": k.model_id,
                "status": k.status,
                "fail_count": k.fail_count,
                "daily_requests": k.daily_request_count,
                "daily_limit": key_limit_map.get(k.api_key_hash),
                # Live rate limit data from API response headers (preferred over static config)
                "ratelimit_limit": k.ratelimit_limit,
                "ratelimit_remaining": k.ratelimit_remaining,
                "cooldown_until": k.cooldown_until.isoformat() if k.cooldown_until else None
            })

        active_models_list = sorted(list(set(k.model_id for k in active_keys)))
        
        return HealthStatus(
            status="operational" if active_keys else "degraded",
            active_keys=len(active_keys),
            cooldown_keys=len(cooldown_keys),
            active_models=len(active_models_list),
            active_model_list=active_models_list,
            total_requests_today=total_today,
            provider_health=provider_health
        )

if __name__ == "__main__":
    import uvicorn
    import sys
    import os

    if "--reset-db" in sys.argv:
        print("Clearing database...")
        if os.path.exists("gateway.db"):
            os.remove("gateway.db")
            print("gateway.db removed successfully.")
        else:
            print("gateway.db not found, nothing to clear.")
        # Remove the argument so uvicorn doesn't complain if it parses args (though it usually doesn't here)
        sys.argv.remove("--reset-db")

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
