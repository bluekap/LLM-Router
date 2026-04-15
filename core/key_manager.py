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

from collections import defaultdict


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
        self.priority_indices = defaultdict(int)

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
        """Syncs the DB with the current keys.json:
        - Inserts any new keys that are not yet in the DB (preserving stats for existing ones).
        - Removes any DB entries whose key_hash is no longer present in keys.json.
        """
        active_hashes = {key.key_hash for key in self.keys}

        async with AsyncSessionLocal() as session:
            # Fetch all currently stored key hashes from DB
            result = await session.execute(select(KeyMetadata))
            db_keys = result.scalars().all()
            db_hashes = {km.api_key_hash for km in db_keys}

            # --- Insert new keys ---
            new_hashes = active_hashes - db_hashes
            added = 0
            for key in self.keys:
                if key.key_hash in new_hashes:
                    session.add(KeyMetadata(
                        provider=key.provider,
                        model_id=key.model_id,
                        api_key_hash=key.key_hash,
                        priority=key.priority
                    ))
                    logger.info(f"[sync] NEW key added to DB: {key.provider} / {key.model_id} (P{key.priority}) ({key.key_hash[:8]})")
                    added += 1

            # --- Update priority for existing keys (in case keys.json changed) ---
            for key in self.keys:
                if key.key_hash in db_hashes:
                    await session.execute(
                        update(KeyMetadata)
                        .where(KeyMetadata.api_key_hash == key.key_hash)
                        .values(priority=key.priority)
                    )

            # --- Remove stale keys ---
            removed_hashes = db_hashes - active_hashes
            removed = 0
            for km in db_keys:
                if km.api_key_hash in removed_hashes:
                    logger.warning(f"[sync] REMOVED stale key from DB: {km.provider} / {km.model_id} ({km.api_key_hash[:8]})")
                    await session.delete(km)
                    removed += 1

            await session.commit()
            logger.info(f"[sync] DB sync complete — {len(self.keys)} active keys | +{added} added | -{removed} removed")

    
    async def get_healthiest_key(self, provider: Optional[str] = None, model_id: Optional[str] = None) -> Optional[Key]:
        """
        Selects the best available API key using a priority-aware round-robin strategy.

        Selection strategy:
        1. Keys are grouped by priority (lower number = higher priority, e.g., P1 before P2).
        2. The system always attempts to select from the highest available priority tier first.
        3. Within a priority tier, keys are distributed using round-robin to ensure fair usage
        across keys and models (prevents a single key/model from being overused).
        4. Within the round-robin group, keys are lightly sorted by `fail_count` to prefer
        healthier keys without sacrificing fairness.
        5. Keys with excessive failures (fail_count >= 5) are temporarily skipped.
        6. Keys in cooldown or inactive state are excluded.
        7. Daily request limits are enforced per key:
        - If the limit is reached and it's the same UTC day → key is skipped.
        - If it's a new UTC day → the counter is reset automatically.

        Additional behavior:
        - If a provider is specified, only keys matching that provider are considered.
        - If a model_id is specified, only keys matching that model_id are considered.
        - Round-robin state is maintained per priority tier using an in-memory index.
        - If no valid keys are available in any priority tier, returns None.

        Args:
            provider (Optional[str]): Optional provider filter (e.g., "openai", "anthropic").
            model_id (Optional[str]): Optional model filter (e.g., "gpt-4", "llama-3-8b").

        Returns:
            Optional[Key]: The selected Key object, or None if no valid key is available.
        """
        async with AsyncSessionLocal() as session:
            now = datetime.datetime.utcnow()
            today_start = datetime.datetime(now.year, now.month, now.day)

            stmt = select(KeyMetadata).where(
                ((KeyMetadata.status == "active") | (KeyMetadata.cooldown_until < now))
            )

            if provider:
                stmt = stmt.where(KeyMetadata.provider == provider)
            
            if model_id:
                stmt = stmt.where(KeyMetadata.model_id == model_id)

            result = await session.execute(stmt)
            healthy_metadata = result.scalars().all()

            if not healthy_metadata:
                return None

            # Map metadata by hash for quick lookup
            meta_map = {m.api_key_hash: m for m in healthy_metadata}

            # Build usable keys grouped by priority
            priority_groups: Dict[int, List[tuple[Key, KeyMetadata]]] = {}

            for key in self.keys:
                meta = meta_map.get(key.key_hash)
                if not meta:
                    continue

                # Daily limit handling
                reset_needed = meta.last_reset_date < today_start
                current_count = 0 if reset_needed else meta.daily_request_count

                if key.daily_limit and current_count >= key.daily_limit:
                    if reset_needed:
                        await session.execute(
                            update(KeyMetadata)
                            .where(KeyMetadata.id == meta.id)
                            .values(
                                daily_request_count=0,
                                last_reset_date=now
                            )
                        )
                    else:
                        continue

                priority_groups.setdefault(meta.priority, []).append((key, meta))

            if not priority_groups:
                return None

            # Iterate priorities (P1 first)
            for priority in sorted(priority_groups.keys()):
                group = priority_groups[priority]

                if not group:
                    continue

                # Sort slightly by health (optional but recommended)
                group.sort(key=lambda x: x[1].fail_count)

                # Round robin starting point
                start_idx = self.priority_indices[priority]

                for i in range(len(group)):
                    idx = (start_idx + i) % len(group)
                    key, meta = group[idx]

                    # Advance pointer for next call
                    self.priority_indices[priority] = (idx + 1) % len(group)

                    return key

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

    async def update_ratelimit_headers(self, key: Key, limit: Optional[int], remaining: Optional[int]):
        """Persists live rate limit data captured from API response headers."""
        if limit is None and remaining is None:
            return
        async with AsyncSessionLocal() as session:
            values = {}
            if limit is not None:
                values["ratelimit_limit"] = limit
            if remaining is not None:
                values["ratelimit_remaining"] = remaining
            stmt = update(KeyMetadata).where(KeyMetadata.api_key_hash == key.key_hash).values(**values)
            await session.execute(stmt)
            await session.commit()
            
    async def get_all_status(self):
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(KeyMetadata))
            return result.scalars().all()
