import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from .database import Base

class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    provider = Column(String)
    model = Column(String)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    latency_ms = Column(Float)
    status_code = Column(Integer)
    error_message = Column(Text, nullable=True)

class KeyMetadata(Base):
    __tablename__ = "key_metadata"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String)
    model_id = Column(String)
    api_key_hash = Column(String, unique=True, index=True) # Hash for identification
    status = Column(String, default="active") # active, cooldown
    fail_count = Column(Integer, default=0)
    cooldown_until = Column(DateTime, nullable=True)
    last_used_timestamp = Column(DateTime, nullable=True)
    daily_request_count = Column(Integer, default=0)
    last_reset_date = Column(DateTime, default=datetime.datetime.utcnow)
