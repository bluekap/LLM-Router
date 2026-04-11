import json
import hashlib
import asyncio
import datetime
import os
import logging
from typing import List, Dict, Optional, Any
from sqlalchemy import select, update
from db.models import KeyMetadata, RequestLog
from db.database import AsyncSessionLocal

logger = logging.getLogger("LLM-Gateway")

class Key:
    def __init__(self, provider: str, model_id: str, api_key: str, priority: int = 1, daily_limit: Optional[int] = None, key_index: int = 0):
        self.provider = provider
        self.model_id = model_id
        self.api_key = api_key
        self.priority = priority
        self.daily_limit = daily_limit
        # Combine provider, model_id, api_key, and an index to guarantee uniqueness even for identical placeholder keys
        hash_input = f"{self.provider}:{self.model_id}:{self.api_key}:{key_index}"
        self.key_hash = hashlib.sha256(hash_input.encode()).hexdigest()
    def __repr__(self):
        return f"Key(provider={self.provider}, model={self.model_id}, hash={self.key_hash[:8]})"

class KeyManager:
    def __init__(self, config_path: str = "keys.json"):
        self.config_path = config_path
        self.keys: List[Key] = []
        self._load_keys()
        self.index = 0 # For round-robin

    def _load_keys(self):
        if not os.path.exists(self.config_path):
            print(f"Warning: {self.config_path} not found.")
            return
            
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
                
                if isinstance(config_data, dict):
                    # Nested structure: { "provider": [ { "model_id": "...", "api_key": "...", ... } ] }
                    all_keys = []
                    for provider, keys_list in config_data.items():
                        for idx, k in enumerate(keys_list):
                            # Ensure provider is set from the key in the dict
                            k_data = k.copy()
                            if 'provider' not in k_data:
                                k_data['provider'] = provider
                            k_data['key_index'] = idx
                            all_keys.append(Key(**k_data))
                    self.keys = all_keys
                elif isinstance(config_data, list):
                    # Old flat list structure: [ { "provider": "...", "model_id": "...", ... } ]
                    self.keys = [Key(key_index=idx, **k) for idx, k in enumerate(config_data)]
                else:
                    raise ValueError("Invalid keys.json format. Expected list or dictionary.")
                    
            print(f"Loaded {len(self.keys)} keys from {self.config_path}")
        except Exception as e:
            print(f"Error loading keys: {e}")

    async def sync_with_db(self):
        """Ensures all loaded keys have entries in the database."""
        async with AsyncSessionLocal() as session:
            for key in self.keys:
                stmt = select(KeyMetadata).where(KeyMetadata.api_key_hash == key.key_hash)
                result = await session.execute(stmt)
                if not result.scalar_one_or_none():
                    metadata = KeyMetadata(
                        provider=key.provider,
                        model_id=key.model_id,
                        api_key_hash=key.key_hash
                    )
                    session.add(metadata)
            await session.commit()
            logger.info(f"Synchronized {len(self.keys)} keys with database.")

    async def get_healthiest_key(self, provider: Optional[str] = None) -> Optional[Key]:
        """
        Implementation of Healthiest selector:
        - Must be Active (not in cooldown)
        - Least fail count (from DB)
        - Least recently used (from DB)
        - Filtered by provider if specified
        """
        async with AsyncSessionLocal() as session:
            # Refresh DB entries for all loaded keys if they don't exist
            # Note: Now primarily handled by sync_with_db on startup

            # Query for healthy keys
            # Filter by cooldown and daily limit
            now = datetime.datetime.utcnow()
            
            # Reset daily counts if it's a new day (UTC)
            today_start = datetime.datetime(now.year, now.month, now.day)
            
            stmt = select(KeyMetadata).where(
                ((KeyMetadata.status == "active") | (KeyMetadata.cooldown_until < now))
            )

            if provider:
                stmt = stmt.where(KeyMetadata.provider == provider)
                
            # Filter logically (we'll do final limit check in python for flexibility)
            result = await session.execute(stmt)
            healthy_metadata = result.scalars().all()
            
            if not healthy_metadata:
                return None
            
            # Check limits and return healthiest
            # Order by fail_count, then last_used
            healthy_metadata.sort(key=lambda x: (x.fail_count, x.last_used_timestamp or datetime.datetime.min))
            
            for meta in healthy_metadata:
                # Find corresponding Key object for limit
                key_obj = next((k for k in self.keys if k.key_hash == meta.api_key_hash), None)
                if not key_obj: continue
                
                # Check daily limit
                reset_needed = meta.last_reset_date < today_start
                current_count = 0 if reset_needed else meta.daily_request_count
                
                if key_obj.daily_limit and current_count >= key_obj.daily_limit:
                    if reset_needed:
                        # Reset if it's a new day
                        await session.execute(update(KeyMetadata).where(KeyMetadata.id == meta.id).values(
                            daily_request_count=0,
                            last_reset_date=now
                        ))
                    else:
                        continue # Skip this key, limit reached
                
                return key_obj
        
        return None

    async def mark_fail(self, key: Key, cooldown_seconds: int = 60):
        """Marks a key as failed and puts it into cooldown."""
        async with AsyncSessionLocal() as session:
            now = datetime.datetime.utcnow()
            cooldown_until = now + datetime.timedelta(seconds=cooldown_seconds)
            
            stmt = update(KeyMetadata).where(KeyMetadata.api_key_hash == key.key_hash).values(
                status="cooldown",
                fail_count=KeyMetadata.fail_count + 1,
                cooldown_until=cooldown_until
            )
            await session.execute(stmt)
            await session.commit()
            print(f"Key {key} marked as COOLDOWN until {cooldown_until}")

    async def reset_fail(self, key: Key):
        """Resets fail count and sets status to active."""
        async with AsyncSessionLocal() as session:
            stmt = update(KeyMetadata).where(KeyMetadata.api_key_hash == key.key_hash).values(
                status="active",
                fail_count=0,
                cooldown_until=None,
                last_used_timestamp=datetime.datetime.utcnow()
            )
            await session.execute(stmt)
            await session.commit()

    async def update_usage(self, key: Key):
        """Updates last used timestamp and daily count."""
        async with AsyncSessionLocal() as session:
            stmt = update(KeyMetadata).where(KeyMetadata.api_key_hash == key.key_hash).values(
                last_used_timestamp=datetime.datetime.utcnow(),
                daily_request_count=KeyMetadata.daily_request_count + 1
            )
            await session.execute(stmt)
            await session.commit()
            
    async def get_all_status(self):
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(KeyMetadata))
            return result.scalars().all()
