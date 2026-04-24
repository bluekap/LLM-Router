#!/usr/bin/env python3
"""
Testing script to test all keys and models concurrently using asyncio.gather.
All keys are tested in parallel for maximum speed.
"""

import asyncio
import json
import datetime
import time
import sys
import os
import logging
from typing import List, Dict, Any

# Silence asyncio internally at the start to reduce clutter
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# Add the parent directory to sys.path so we can import from main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import key_manager, gateway
from schemas import ChatCompletionRequest


async def test_key(key_obj: Any, model_id: str, test_message: str = "hi") -> Dict[str, Any]:
    """Test a single key/model pair and return a result dict."""
    result = {
        "model": model_id,
        "provider": key_obj.provider,
        "key_hash": key_obj.key_hash[:8],
        "status": "pending",
        "response": None,
        "error": None,
        "latency_ms": None,
        "tokens_used": None,
    }

    try:
        request = ChatCompletionRequest(
            model=model_id,
            messages=[{"role": "user", "content": test_message}],
            temperature=0.0,
            max_tokens=10,
        )

        start_time = time.time()
        response = await gateway.chat_completion(request, forced_key=key_obj)
        latency = (time.time() - start_time) * 1000

        result["status"] = "success"
        result["latency_ms"] = round(latency, 2)
        result["response"] = str(response)[:100] if response else None

        if hasattr(response, "usage"):
            result["tokens_used"] = getattr(response.usage, "total_tokens", 0)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    icon = "✅" if result["status"] == "success" else "❌"
    lat = f"{result['latency_ms']:.0f}ms" if result["latency_ms"] else "-"
    print(f"  {icon} [{key_obj.provider}] {model_id} ({key_obj.key_hash[:8]}) — {lat}")
    if result["error"]:
        # Print a short inline error hint; full error goes to the log file
        hint = str(result["error"]).replace("\n", " ")[:120]
        print(f"       └─ {hint}")

    return result


async def test_all_keys():
    """Collect all key/model pairs and test them all concurrently."""

    # Suppress SSL teardown noise on shutdown
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(
        lambda loop, ctx: None
        if any(kw in ctx.get("message", "") for kw in ("Fatal write error", "Event loop is closed"))
        else None
    )

    print("🧪 Testing all keys and models (parallel)...")
    print("=" * 60)

    all_keys_meta = await key_manager.get_all_status()

    if not all_keys_meta:
        print("❌ No keys found in database")
        return

    # Map DB records → loaded Key objects (which carry the actual api_key)
    key_map = {k.key_hash: k for k in key_manager.keys}

    # Build flat list of (key_obj, model_id) pairs to test
    tasks_meta = []
    for key_meta in all_keys_meta:
        key_obj = key_map.get(key_meta.api_key_hash)
        if not key_obj:
            print(f"⚠️  Skipping {key_meta.api_key_hash[:8]} — not in current keys.json")
            continue
        tasks_meta.append((key_obj, key_meta.model_id))

    print(f"📊 Launching {len(tasks_meta)} concurrent tests across {len(set(m for _, m in tasks_meta))} models")
    print()

    # Fire all tests in parallel
    start_all = time.time()
    results: List[Dict[str, Any]] = await asyncio.gather(
        *[test_key(key_obj, model_id) for key_obj, model_id in tasks_meta],
        return_exceptions=False,
    )
    total_wall = (time.time() - start_all) * 1000

    # ── Tabular summary ──────────────────────────────────────────────────────
    sorted_results = sorted(results, key=lambda x: (x["provider"], x["model"]))

    header = f"{'Model (Provider)':<48} | {'Hits':<5} | {'Misses':<6} | {'Avg Lat':<8} | Status"
    separator = "─" * len(header)

    grouped: Dict[str, Dict] = {}
    for r in sorted_results:
        k = f"{r['model']} ({r['provider']})"
        g = grouped.setdefault(k, {"hits": 0, "misses": 0, "latencies": [], "errors": []})
        if r["status"] == "success":
            g["hits"] += 1
            if r["latency_ms"]:
                g["latencies"].append(r["latency_ms"])
        else:
            g["misses"] += 1
            if r["error"]:
                g["errors"].append(r["error"])

    print()
    print("═" * len(header))
    print("📊  DIAGNOSTIC SUMMARY")
    print("═" * len(header))
    print(header)
    print(separator)

    for model_key, stats in grouped.items():
        hits = stats["hits"]
        misses = stats["misses"]
        avg_lat = (
            f"{sum(stats['latencies'])/len(stats['latencies']):.0f}ms"
            if stats["latencies"]
            else "—"
        )
        icon = "✅" if misses == 0 else ("❌" if hits == 0 else "⚠️ ")
        print(f"{model_key:<48} | {hits:<5} | {misses:<6} | {avg_lat:<8} | {icon}")

        # Print each unique full error under the row
        for err in dict.fromkeys(stats["errors"]):  # deduplicated, order-preserving
            clean = err.replace("\n", " ").strip()
            print(f"   └─ {clean}")

    print("═" * len(header))

    successes = [r for r in results if r["status"] == "success"]
    failures  = [r for r in results if r["status"] == "failed"]
    total     = len(results)

    print(f"OVERALL : {len(successes)} ✅  /  {len(failures)} ❌  ({total} total)")
    if successes:
        g_avg = sum(r["latency_ms"] for r in successes) / len(successes)
        print(f"AVG LAT : {g_avg:.0f}ms   |   WALL TIME: {total_wall:.0f}ms")

    # ── Write full JSON log ──────────────────────────────────────────────────
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    run_ts   = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"test_run_{run_ts}.json")

    report = {
        "timestamp": run_ts,
        "wall_time_ms": round(total_wall, 2),
        "total": total,
        "success": len(successes),
        "failed": len(failures),
        "global_avg_latency_ms": round(g_avg, 2) if successes else None,
        "results": sorted_results,
    }
    with open(log_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📁 Full report → {log_path}")

    # Give background SSL connections a moment to settle cleanly
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.wait(pending, timeout=1.5)
    await asyncio.sleep(0.3)

    return results


if __name__ == "__main__":
    try:
        asyncio.run(test_all_keys())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}")