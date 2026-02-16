"""Basic smoke tests for revvel-forensic-mcp."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_imports():
    from src.face_detection import FaceDetector, MaskDetector
    from src.beauty_enhancement import BeautyEnhancer
    from src.forensic_analysis import ForensicAnalyzer
    assert FaceDetector is not None
    assert BeautyEnhancer is not None
    assert ForensicAnalyzer is not None


def test_mcp_server_initialize():
    from server import MCPServer
    srv = MCPServer()
    resp = srv.handle_initialize(1, {"protocolVersion": "2024-11-05"})
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == "revvel-forensic-mcp"
    assert resp["result"]["serverInfo"]["version"] == "2.0.0"


def test_mcp_tools_list():
    from server import MCPServer
    srv = MCPServer()
    resp = srv.handle_tools_list(2, {})
    tools = resp["result"]["tools"]
    names = [t["name"] for t in tools]
    assert "enhance_image" in names
    assert "detect_mask" in names
    assert "full_analysis" in names
    assert "list_models" in names
    assert "set_model" in names


def test_model_adapter():
    from server import ModelAdapter
    ma = ModelAdapter()
    providers = ma.list_providers()
    assert len(providers) > 0
    assert ma.set_provider("openai")
    assert not ma.set_provider("nonexistent")


def test_stream_emitter():
    from server import StreamEmitter
    se = StreamEmitter()
    se.register("job1")
    se.emit("job1", "progress", {"pct": 50})
    events = se.get_events("job1")
    assert len(events) == 1
    assert events[0]["event"] == "progress"
    se.cleanup("job1")


def test_agent_integration():
    from server import AgentIntegration
    ai = AgentIntegration()
    lc = ai.langchain_tools()
    assert len(lc) > 0
    oai = ai.openai_function_defs()
    assert len(oai) > 0
    assert oai[0]["type"] == "function"
