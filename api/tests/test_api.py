"""Basic smoke tests for revvel-forensic-api."""

import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_imports():
    from src.face_detection import FaceDetector, MaskDetector
    from src.beauty_enhancement import BeautyEnhancer
    from src.forensic_analysis import ForensicAnalyzer
    assert FaceDetector is not None
    assert BeautyEnhancer is not None
    assert ForensicAnalyzer is not None


def test_app_creation():
    from app import app
    assert app.title == "Revvel Forensic API"
    assert app.version == "2.0.0"


def test_key_store():
    from app import APIKeyStore
    store = APIKeyStore()
    key = store.create("test", "user")
    assert store.validate(key)
    store.revoke(key)
    assert not store.validate(key)


def test_rate_limiter():
    from app import RateLimiter
    rl = RateLimiter(requests_per_minute=5)
    for _ in range(5):
        assert rl.check("test-key")
    assert not rl.check("test-key")
