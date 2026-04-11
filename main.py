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
        
        provider_health = {}
        for k in all_keys:
            if k.provider not in provider_health:
                provider_health[k.provider] = {"active": 0, "cooldown": 0, "fail_count": 0}
            
            if k.status == "active":
                provider_health[k.provider]["active"] += 1
            else:
                provider_health[k.provider]["cooldown"] += 1
            provider_health[k.provider]["fail_count"] += k.fail_count

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
