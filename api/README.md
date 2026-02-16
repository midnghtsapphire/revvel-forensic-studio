# Revvel Forensic API

Standalone REST API server for forensic image analysis and beauty enhancement. Split from the `revvel-forensic-studio` monorepo and enhanced with Blue Ocean features.

## Features

### Core Endpoints
- **POST /enhance** — Apply beauty enhancements (natural, glamour, dramatic, subtle)
- **POST /batch-enhance** — Batch-enhance multiple images
- **POST /detect-mask** — Detect if a face is wearing a mask
- **POST /reconstruct-face** — Reconstruct face from masked image
- **POST /analyze-exif** — Extract and analyse EXIF metadata
- **POST /detect-objects** — Detect glasses, jewelry, tags, tattoos
- **POST /decompose-layers** — Frequency-based layer decomposition
- **POST /enhance-edges** — CLAHE + Canny edge enhancement
- **POST /zoom-enhance** — Region zoom with detail recovery
- **POST /full-analysis** — Comprehensive forensic analysis

### Blue Ocean Enhancements
- **Swagger / OpenAPI Docs** — Interactive docs at `/docs` and `/redoc`
- **API Key Management** — Create, list, and revoke API keys via `/keys`
- **Rate Limiting** — Configurable per-minute rate limit per key
- **Webhook Callbacks** — Async POST callbacks on job completion (with HMAC signatures)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Example Requests

```bash
# Enhance an image (no auth in dev mode)
curl -X POST http://localhost:8000/enhance \
  -F "file=@photo.jpg" -F "preset=glamour" -o enhanced.jpg

# Detect mask with webhook callback
curl -X POST http://localhost:8000/detect-mask \
  -H "X-API-Key: revvel-master-key" \
  -F "file=@suspect.jpg" \
  -F "webhook_url=https://example.com/hook" | jq .

# Create a new API key
curl -X POST http://localhost:8000/keys \
  -H "X-API-Key: revvel-master-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app", "role": "user"}'

# Check rate limit status
curl http://localhost:8000/rate-limit-status \
  -H "X-API-Key: revvel-master-key"
```

## Docker

```bash
docker build -t revvel-forensic-api .
docker run --rm -p 8000:8000 -e AUTH_DISABLED=true revvel-forensic-api
```

## Configuration

See `.env.example` for all environment variables.

| Variable | Default | Description |
|---|---|---|
| `PORT` | 8000 | Server port |
| `API_MASTER_KEY` | revvel-master-key | Admin API key |
| `AUTH_DISABLED` | false | Disable auth for dev |
| `RATE_LIMIT_RPM` | 120 | Requests per minute |
| `FACE_BACKEND` | auto | Face detection backend |

## Architecture

```
revvel-forensic-api/
├── app.py                  # FastAPI application
├── src/
│   ├── __init__.py
│   ├── face_detection.py   # Face detection & mask detection
│   ├── beauty_enhancement.py  # Beauty/makeup engine
│   └── forensic_analysis.py   # Forensic tools
├── tests/
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT License

## Credits

Developed by Revvel AI Engine — split from `revvel-forensic-studio` by Team 4.
