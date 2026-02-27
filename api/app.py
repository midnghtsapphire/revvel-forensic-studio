#!/usr/bin/env python3
"""
Revvel Forensic API — Standalone REST API server for forensic analysis.

Blue Ocean enhancements:
  - Webhook callbacks for async job completion
  - Rate limiting per API key
  - API key management (create / revoke / list)
  - Full Swagger / OpenAPI docs (built-in via FastAPI)
"""

import os
import sys
import json
import uuid
import time
import hmac
import hashlib
import asyncio
import stripe
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Header, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests as http_requests

from src.face_detection import FaceDetector, MaskDetector
from src.beauty_enhancement import BeautyEnhancer
from src.forensic_analysis import (
    ForensicAnalyzer, FaceReconstructor, EXIFAnalyzer,
    ObjectDetector, LayerAnalyzer, EdgeEnhancer,
)

logger = logging.getLogger("revvel-forensic-api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Blue Ocean: In-memory API key store (swap for DB in production)
# ---------------------------------------------------------------------------

class APIKeyStore:
    """Simple in-memory API key manager."""

    def __init__(self):
        self._keys: Dict[str, Dict[str, Any]] = {}
        # Seed a default master key from env
        master = os.getenv("API_MASTER_KEY", "revvel-master-key")
        self._keys[master] = {
            "name": "master",
            "created": datetime.utcnow().isoformat(),
            "role": "admin",
            "active": True,
        }

    def create(self, name: str, role: str = "user") -> str:
        key = f"rvl-{uuid.uuid4().hex[:24]}"
        self._keys[key] = {
            "name": name,
            "created": datetime.utcnow().isoformat(),
            "role": role,
            "active": True,
        }
        return key

    def revoke(self, key: str) -> bool:
        if key in self._keys:
            self._keys[key]["active"] = False
            return True
        return False

    def validate(self, key: str) -> bool:
        return key in self._keys and self._keys[key]["active"]

    def list_keys(self) -> List[Dict]:
        return [
            {"key": k[:8] + "...", **v}
            for k, v in self._keys.items()
        ]

    def get_role(self, key: str) -> str:
        return self._keys.get(key, {}).get("role", "user")


key_store = APIKeyStore()


# ---------------------------------------------------------------------------
# Blue Ocean: Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter keyed by API key."""

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self._buckets: Dict[str, list] = defaultdict(list)

    def check(self, api_key: str) -> bool:
        now = time.time()
        window = now - 60
        self._buckets[api_key] = [t for t in self._buckets[api_key] if t > window]
        if len(self._buckets[api_key]) >= self.rpm:
            return False
        self._buckets[api_key].append(now)
        return True

    def remaining(self, api_key: str) -> int:
        now = time.time()
        window = now - 60
        self._buckets[api_key] = [t for t in self._buckets[api_key] if t > window]
        return max(0, self.rpm - len(self._buckets[api_key]))


rate_limiter = RateLimiter(requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "120")))


# ---------------------------------------------------------------------------
# Blue Ocean: Stripe Billing
# ---------------------------------------------------------------------------

class BillingManager:
    """Manage Stripe billing in test and live modes."""

    def __init__(self):
        self.mode = os.getenv("STRIPE_MODE", "test").lower()
        if self.mode == "live":
            self.secret_key = os.getenv("STRIPE_LIVE_SECRET_KEY")
            self.publishable_key = os.getenv("STRIPE_LIVE_PUBLISHABLE_KEY")
        else:
            self.secret_key = os.getenv("STRIPE_TEST_SECRET_KEY")
            self.publishable_key = os.getenv("STRIPE_TEST_PUBLISHABLE_KEY")
        
        if self.secret_key:
            stripe.api_key = self.secret_key

    def get_config(self):
        return {
            "mode": self.mode,
            "publishable_key": self.publishable_key,
            "is_configured": bool(self.secret_key)
        }

billing_manager = BillingManager()


# ---------------------------------------------------------------------------
# Blue Ocean: Webhook dispatcher
# ---------------------------------------------------------------------------

class WebhookDispatcher:
    """Fire-and-forget webhook callbacks."""

    @staticmethod
    async def send(url: str, payload: Dict[str, Any], secret: Optional[str] = None):
        """POST *payload* to *url* with optional HMAC signature."""
        try:
            headers = {"Content-Type": "application/json"}
            body = json.dumps(payload, default=str)
            if secret:
                sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
                headers["X-Revvel-Signature"] = sig
            http_requests.post(url, data=body, headers=headers, timeout=10)
            logger.info(f"Webhook delivered to {url}")
        except Exception as exc:
            logger.warning(f"Webhook failed for {url}: {exc}")


webhook = WebhookDispatcher()


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Validate the X-API-Key header."""
    # Allow unauthenticated access if env says so (dev mode)
    if os.getenv("AUTH_DISABLED", "false").lower() == "true":
        return "anonymous"
    if not x_api_key or not key_store.validate(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if not rate_limiter.check(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return x_api_key


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Revvel Forensic API",
    description=(
        "REST API for image forensic analysis, beauty enhancement, "
        "and face detection. Includes webhook callbacks, rate limiting, "
        "and API key management."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise core engines once
face_detector = FaceDetector()
mask_detector = MaskDetector()
beauty_enhancer = BeautyEnhancer()
forensic_analyzer = ForensicAnalyzer()
face_reconstructor = FaceReconstructor()
exif_analyzer = EXIFAnalyzer()
object_detector = ObjectDetector()
layer_analyzer = LayerAnalyzer()
edge_enhancer = EdgeEnhancer()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class KeyCreateRequest(BaseModel):
    name: str = Field(..., description="Friendly name for the key")
    role: str = Field("user", description="Role: user or admin")

class WebhookRequest(BaseModel):
    url: str = Field(..., description="Callback URL")
    secret: Optional[str] = Field(None, description="HMAC secret for signature")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "img.jpg").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        return tmp.name


def _load_upload(upload: UploadFile) -> np.ndarray:
    path = _save_upload(upload)
    img = cv2.imread(path)
    os.unlink(path)
    if img is None:
        raise HTTPException(400, "Cannot decode image")
    return img


def _save_temp(image: np.ndarray, suffix: str = ".jpg") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        cv2.imwrite(tmp.name, image)
        return tmp.name


def _np_convert(obj):
    """Recursively convert numpy types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _np_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_np_convert(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Routes — API Key Management (Blue Ocean)
# ---------------------------------------------------------------------------

@app.post("/keys", tags=["API Keys"])
async def create_api_key(body: KeyCreateRequest, api_key: str = Depends(verify_api_key)):
    """Create a new API key (admin only)."""
    if key_store.get_role(api_key) != "admin" and api_key != "anonymous":
        raise HTTPException(403, "Admin role required")
    new_key = key_store.create(body.name, body.role)
    return {"api_key": new_key, "name": body.name, "role": body.role}


@app.delete("/keys/{key_prefix}", tags=["API Keys"])
async def revoke_api_key(key_prefix: str, api_key: str = Depends(verify_api_key)):
    """Revoke an API key by prefix (admin only)."""
    if key_store.get_role(api_key) != "admin" and api_key != "anonymous":
        raise HTTPException(403, "Admin role required")
    # Find full key by prefix
    for k in list(key_store._keys.keys()):
        if k.startswith(key_prefix):
            key_store.revoke(k)
            return {"revoked": True}
    raise HTTPException(404, "Key not found")


@app.get("/keys", tags=["API Keys"])
async def list_api_keys(api_key: str = Depends(verify_api_key)):
    """List all API keys (masked)."""
    return {"keys": key_store.list_keys()}


# ---------------------------------------------------------------------------
# Routes — Beauty Enhancement
# ---------------------------------------------------------------------------

@app.post("/enhance", tags=["Beauty"])
async def enhance_image(
    file: UploadFile = File(...),
    preset: str = Form("natural"),
    skin_smooth: Optional[float] = Form(None),
    brightness: Optional[float] = Form(None),
    webhook_url: Optional[str] = Form(None),
    webhook_secret: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(verify_api_key),
):
    """Apply beauty enhancements to an image."""
    image = _load_upload(file)
    custom = {}
    if skin_smooth is not None:
        custom["skin_smooth"] = skin_smooth
    if brightness is not None:
        custom["brightness"] = brightness

    enhanced = beauty_enhancer.enhance(image, preset=preset,
                                       custom_params=custom if custom else None)
    output_path = _save_temp(enhanced)

    if webhook_url:
        background_tasks.add_task(
            webhook.send, webhook_url,
            {"event": "enhance_complete", "preset": preset, "output": output_path},
            webhook_secret,
        )

    return FileResponse(output_path, media_type="image/jpeg",
                        filename=f"enhanced_{file.filename}")


@app.post("/batch-enhance", tags=["Beauty"])
async def batch_enhance(
    files: List[UploadFile] = File(...),
    preset: str = Form("natural"),
    api_key: str = Depends(verify_api_key),
):
    """Batch-enhance multiple images."""
    results = []
    for f in files:
        img = _load_upload(f)
        enh = beauty_enhancer.enhance(img, preset=preset)
        out = _save_temp(enh)
        results.append({"filename": f.filename, "output_path": out})
    return {"status": "success", "processed": len(results), "results": results}


# ---------------------------------------------------------------------------
# Routes — Forensic Analysis
# ---------------------------------------------------------------------------

@app.post("/detect-mask", tags=["Forensics"])
async def detect_mask(
    file: UploadFile = File(...),
    webhook_url: Optional[str] = Form(None),
    webhook_secret: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(verify_api_key),
):
    """Detect if a face is wearing a mask."""
    image = _load_upload(file)
    result = mask_detector.detect_mask(image)
    if webhook_url:
        background_tasks.add_task(webhook.send, webhook_url, {"event": "mask_detection", **result}, webhook_secret)
    return JSONResponse(content=_np_convert(result))


@app.post("/reconstruct-face", tags=["Forensics"])
async def reconstruct_face(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    """Reconstruct face from masked/obscured image."""
    image = _load_upload(file)
    result = face_reconstructor.reconstruct_from_masked(image)
    if result["success"]:
        out = _save_temp(result["reconstructed"])
        return FileResponse(out, media_type="image/jpeg", filename=f"reconstructed_{file.filename}")
    return JSONResponse(content={"success": False, "reason": result["reason"]})


@app.post("/analyze-exif", tags=["Forensics"])
async def analyze_exif(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    """Analyse EXIF metadata."""
    path = _save_upload(file)
    result = exif_analyzer.analyze(path)
    os.unlink(path)
    return JSONResponse(content=_np_convert(result))


@app.post("/detect-objects", tags=["Forensics"])
async def detect_objects(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    """Detect objects (glasses, jewelry, tags, tattoos)."""
    image = _load_upload(file)
    result = object_detector.detect_all(image)
    return JSONResponse(content=_np_convert(result))


@app.post("/decompose-layers", tags=["Forensics"])
async def decompose_layers(
    file: UploadFile = File(...),
    num_layers: int = Form(5),
    api_key: str = Depends(verify_api_key),
):
    """Decompose image into frequency layers."""
    image = _load_upload(file)
    layers = layer_analyzer.decompose(image, num_layers=num_layers)
    paths = []
    for i, layer in enumerate(layers):
        p = _save_temp(layer, suffix=f"_layer_{i}.png")
        paths.append(p)
    return {"status": "success", "num_layers": len(layers), "layer_paths": paths}


@app.post("/enhance-edges", tags=["Forensics"])
async def enhance_edges(
    file: UploadFile = File(...),
    strength: float = Form(1.5),
    api_key: str = Depends(verify_api_key),
):
    """Enhance edges using lighting and makeup patterns."""
    image = _load_upload(file)
    enhanced = edge_enhancer.enhance_facial_edges(image, strength=strength)
    out = _save_temp(enhanced)
    return FileResponse(out, media_type="image/jpeg", filename=f"edges_{file.filename}")


@app.post("/zoom-enhance", tags=["Forensics"])
async def zoom_enhance(
    file: UploadFile = File(...),
    x: int = Form(...), y: int = Form(...),
    width: int = Form(...), height: int = Form(...),
    scale: float = Form(2.0),
    api_key: str = Depends(verify_api_key),
):
    """Zoom into a region and enhance details."""
    image = _load_upload(file)
    enhanced = edge_enhancer.zoom_and_enhance(image, (x, y, width, height), scale=scale)
    out = _save_temp(enhanced)
    return FileResponse(out, media_type="image/jpeg", filename=f"zoomed_{file.filename}")


@app.post("/full-analysis", tags=["Forensics"])
async def full_analysis(
    file: UploadFile = File(...),
    webhook_url: Optional[str] = Form(None),
    webhook_secret: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(verify_api_key),
):
    """Perform full forensic analysis."""
    path = _save_upload(file)
    result = forensic_analyzer.full_analysis(path)
    os.unlink(path)
    result = _np_convert(result)
    if webhook_url:
        background_tasks.add_task(webhook.send, webhook_url, {"event": "full_analysis", **result}, webhook_secret)
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Routes — Utility
# ---------------------------------------------------------------------------

@app.get("/", tags=["Utility"], response_class=HTMLResponse)
async def root():
    """Serve the Forensic Studio UI landing page."""
    static_index = Path(__file__).parent / "static" / "index.html"
    if static_index.exists():
        return HTMLResponse(content=static_index.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Revvel Forensic Studio API v2.0.0</h1>", status_code=200)


@app.get("/health", tags=["Utility"])
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/rate-limit-status", tags=["Utility"])
async def rate_limit_status(api_key: str = Depends(verify_api_key)):
    """Check remaining rate-limit quota."""
    return {"remaining": rate_limiter.remaining(api_key), "limit_per_minute": rate_limiter.rpm}


@app.get("/billing/config", tags=["Billing"])
async def get_billing_config(api_key: str = Depends(verify_api_key)):
    """Get public billing configuration (Blue Ocean)."""
    return billing_manager.get_config()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
