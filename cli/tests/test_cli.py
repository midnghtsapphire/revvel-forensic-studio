"""Basic smoke tests for revvel-forensic-cli."""

import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "cli.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Revvel Forensic CLI" in result.stdout


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "cli.py", "--version"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "2.0.0" in result.stdout


def test_imports():
    from src.face_detection import FaceDetector, MaskDetector
    from src.beauty_enhancement import BeautyEnhancer
    from src.forensic_analysis import ForensicAnalyzer
    assert FaceDetector is not None
    assert MaskDetector is not None
    assert BeautyEnhancer is not None
    assert ForensicAnalyzer is not None
