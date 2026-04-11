# LLM Router Gateway 🚀

A production-grade Python FastAPI gateway that provides a unified, OpenAI-compatible endpoint for interacting with multiple LLM providers. Designed for high availability, it manages a pool of free-tier/paid API keys, implements smart round-robin load balancing, automatic waterfall failover, and rate-limit tracking.

## Features ✨

- **Unified OpenAI Endpoint**: Talk to Groq, Gemini, Mistral, Cerebras, and OpenRouter using the standard OpenAI `/v1/chat/completions` payload.
- **Smart Key Rotation**: Round-robin selection picks the "healthiest" key (least failed, least recently used).
- **Waterfall Failover**: Automatically catches `429 Rate Limit` and `5xx Server Error` responses, puts the failing key in cooldown, and immediately retries with the next available key.
- **Daily Quotas**: Built-in support for provider-specific daily request limits (e.g., Gemini's 100 requests/day).
- **Observability**: Asynchronously logs all requests (tokens, latency, provider, errors) to a local SQLite database (`gateway.db`).
- **Health Dashboard**: Real-time monitoring of active keys, cooldowns, and daily request volumes.

## Supported Providers & Models

Powered by [LiteLLM](https://github.com/BerriAI/litellm), this gateway inherently supports 100+ LLMs. The current configuration is optimized for:
- **Groq** (Llama 3 70B/8B)
- **Google Gemini** (Gemini 3 Flash)
- **Mistral AI** (Mistral Large)
- **OpenRouter** (Free tier fallback)
- **Cerebras** (Ultra-fast Llama 3.1)

## Setup & Installation 🛠️

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd LLM-Router
   ```

2. **Set up the Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure API Keys:**
   Copy the example config and fill in your actual API keys:
   ```bash
   cp keys.json.example keys.json
   ```
   *Note: `keys.json` is ignored by Git to protect your credentials.*

4. **Run the Gateway:**
   ```bash
   python main.py
   ```
   The gateway will start on `http://localhost:8000`.

## Architecture Overview 🏗️

- **`main.py`**: The FastAPI application and core routing endpoints.
- **`core/key_manager.py`**: Handles loading keys, checking database health, tracking daily limits, and smart selection logic.
- **`core/gateway.py`**: Wraps LiteLLM to handle the actual LLM requests and manages the waterfall retry loop.
- **`db/`**: Contains asynchronous SQLAlchemy SQLite configuration (`database.py`) and schema (`models.py`) for logging.

## Usage 💻

### 1. Chat Completions
Send an OpenAI-compatible request to the gateway. If you omit the `"model"` field, the gateway will automatically map your request to the healthiest available key and its default model.

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [{"role": "user", "content": "Explain quantum entanglement in one sentence."}]
     }'
```

### 2. View Health Dashboard
Open your browser to `http://localhost:8000/` or `http://localhost:8000/health` to view the real-time status of your API keys and today's usage metrics.

## Contributing
Feel free to open issues or submit pull requests for enhancements, additional provider optimizations, or bug fixes.
