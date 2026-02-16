#!/usr/bin/env python3
"""
Revvel Forensic MCP Server — Standalone Model Context Protocol server
for AI agent integration with forensic analysis capabilities.

Blue Ocean enhancements:
  - Streaming results via SSE / chunked responses
  - Multi-model support (OpenAI, Anthropic, Google, local)
  - Integration helpers for popular AI agents (LangChain, CrewAI, AutoGen)
  - Progress callbacks during long-running operations
"""

import os
import sys
import json
import uuid
import time
import asyncio
import tempfile
import base64
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime

import cv2
import numpy as np

from src.face_detection import FaceDetector, MaskDetector, load_image, save_image
from src.beauty_enhancement import BeautyEnhancer
from src.forensic_analysis import (
    ForensicAnalyzer, FaceReconstructor, EXIFAnalyzer,
    ObjectDetector, LayerAnalyzer, EdgeEnhancer,
)

logger = logging.getLogger("revvel-forensic-mcp")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Blue Ocean: Multi-model adapter
# ---------------------------------------------------------------------------

class ModelAdapter:
    """Unified interface for multiple AI model providers."""

    SUPPORTED_PROVIDERS = ["openai", "anthropic", "google", "local", "openrouter"]

    def __init__(self):
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._active_provider: str = os.getenv("MCP_MODEL_PROVIDER", "openai")

        # Auto-configure from env
        if os.getenv("OPENAI_API_KEY"):
            self._configs["openai"] = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            }
        if os.getenv("ANTHROPIC_API_KEY"):
            self._configs["anthropic"] = {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            }
        if os.getenv("GOOGLE_API_KEY"):
            self._configs["google"] = {
                "api_key": os.getenv("GOOGLE_API_KEY"),
                "model": os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            }
        if os.getenv("OPENROUTER_API_KEY"):
            self._configs["openrouter"] = {
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "model": os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-20250514"),
                "base_url": "https://openrouter.ai/api/v1",
            }

    @property
    def active_provider(self) -> str:
        return self._active_provider

    def set_provider(self, provider: str) -> bool:
        if provider in self.SUPPORTED_PROVIDERS:
            self._active_provider = provider
            return True
        return False

    def list_providers(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": p,
                "configured": p in self._configs,
                "active": p == self._active_provider,
            }
            for p in self.SUPPORTED_PROVIDERS
        ]

    def get_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        p = provider or self._active_provider
        return self._configs.get(p, {})


model_adapter = ModelAdapter()


# ---------------------------------------------------------------------------
# Blue Ocean: Streaming result emitter
# ---------------------------------------------------------------------------

class StreamEmitter:
    """Emit streaming progress and partial results."""

    def __init__(self):
        self._listeners: Dict[str, List] = {}

    def register(self, job_id: str):
        self._listeners[job_id] = []

    def emit(self, job_id: str, event: str, data: Any):
        entry = {"event": event, "data": data, "timestamp": datetime.utcnow().isoformat()}
        if job_id in self._listeners:
            self._listeners[job_id].append(entry)
        logger.debug(f"Stream [{job_id}] {event}: {data}")

    def get_events(self, job_id: str, after: int = 0) -> List[Dict]:
        events = self._listeners.get(job_id, [])
        return events[after:]

    def cleanup(self, job_id: str):
        self._listeners.pop(job_id, None)

    async def stream(self, job_id: str) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted events as they arrive."""
        idx = 0
        while True:
            events = self.get_events(job_id, after=idx)
            for ev in events:
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], default=str)}\n\n"
                idx += 1
                if ev["event"] == "complete":
                    return
            await asyncio.sleep(0.1)


stream_emitter = StreamEmitter()


# ---------------------------------------------------------------------------
# Core forensic engine wrappers
# ---------------------------------------------------------------------------

face_detector = FaceDetector()
mask_detector = MaskDetector()
beauty_enhancer = BeautyEnhancer()
forensic_analyzer = ForensicAnalyzer()
face_reconstructor = FaceReconstructor()
exif_analyzer = EXIFAnalyzer()
object_detector = ObjectDetector()
layer_analyzer = LayerAnalyzer()
edge_enhancer = EdgeEnhancer()


def _np_convert(obj):
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


def _image_to_base64(image: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", image)
    return base64.b64encode(buf).decode("utf-8")


def _base64_to_image(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# MCP Tool definitions
# ---------------------------------------------------------------------------

MCP_TOOLS = {
    "enhance_image": {
        "description": "Apply beauty enhancements to an image",
        "parameters": {
            "image_base64": {"type": "string", "description": "Base64-encoded image"},
            "image_path": {"type": "string", "description": "Path to image file (alternative to base64)"},
            "preset": {"type": "string", "enum": ["natural", "glamour", "dramatic", "subtle"], "default": "natural"},
            "skin_smooth": {"type": "number", "description": "Skin smoothing 0-1"},
            "brightness": {"type": "number", "description": "Brightness 0.8-1.2"},
        },
    },
    "detect_mask": {
        "description": "Detect if a face is wearing a mask",
        "parameters": {
            "image_base64": {"type": "string"},
            "image_path": {"type": "string"},
        },
    },
    "reconstruct_face": {
        "description": "Reconstruct face from masked/obscured image",
        "parameters": {
            "image_base64": {"type": "string"},
            "image_path": {"type": "string"},
        },
    },
    "analyze_exif": {
        "description": "Analyse EXIF metadata of an image",
        "parameters": {
            "image_path": {"type": "string", "description": "Path to image file"},
        },
    },
    "detect_objects": {
        "description": "Detect objects (glasses, jewelry, tags, tattoos)",
        "parameters": {
            "image_base64": {"type": "string"},
            "image_path": {"type": "string"},
        },
    },
    "decompose_layers": {
        "description": "Decompose image into frequency layers",
        "parameters": {
            "image_base64": {"type": "string"},
            "image_path": {"type": "string"},
            "num_layers": {"type": "integer", "default": 5},
        },
    },
    "enhance_edges": {
        "description": "Enhance edges using lighting and makeup patterns",
        "parameters": {
            "image_base64": {"type": "string"},
            "image_path": {"type": "string"},
            "strength": {"type": "number", "default": 1.5},
        },
    },
    "zoom_enhance": {
        "description": "Zoom into a region and enhance details",
        "parameters": {
            "image_base64": {"type": "string"},
            "image_path": {"type": "string"},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
            "scale": {"type": "number", "default": 2.0},
        },
    },
    "full_analysis": {
        "description": "Perform comprehensive forensic analysis on an image",
        "parameters": {
            "image_path": {"type": "string", "description": "Path to image file"},
            "stream": {"type": "boolean", "default": False, "description": "Enable streaming progress (Blue Ocean)"},
        },
    },
    "list_models": {
        "description": "List available AI model providers (Blue Ocean)",
        "parameters": {},
    },
    "set_model": {
        "description": "Switch active AI model provider (Blue Ocean)",
        "parameters": {
            "provider": {"type": "string", "enum": ["openai", "anthropic", "google", "local", "openrouter"]},
        },
    },
}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _load_image(params: Dict) -> np.ndarray:
    if params.get("image_base64"):
        return _base64_to_image(params["image_base64"])
    if params.get("image_path"):
        img = cv2.imread(params["image_path"])
        if img is None:
            raise ValueError(f"Cannot load image: {params['image_path']}")
        return img
    raise ValueError("Provide image_base64 or image_path")


def handle_enhance_image(params: Dict) -> Dict:
    image = _load_image(params)
    custom = {}
    if params.get("skin_smooth") is not None:
        custom["skin_smooth"] = params["skin_smooth"]
    if params.get("brightness") is not None:
        custom["brightness"] = params["brightness"]
    enhanced = beauty_enhancer.enhance(
        image, preset=params.get("preset", "natural"),
        custom_params=custom if custom else None,
    )
    return {"success": True, "image_base64": _image_to_base64(enhanced)}


def handle_detect_mask(params: Dict) -> Dict:
    image = _load_image(params)
    result = mask_detector.detect_mask(image)
    return _np_convert(result)


def handle_reconstruct_face(params: Dict) -> Dict:
    image = _load_image(params)
    result = face_reconstructor.reconstruct_from_masked(image)
    if result["success"]:
        return {
            "success": True,
            "confidence": result["confidence"],
            "image_base64": _image_to_base64(result["reconstructed"]),
        }
    return {"success": False, "reason": result["reason"]}


def handle_analyze_exif(params: Dict) -> Dict:
    path = params.get("image_path")
    if not path:
        raise ValueError("image_path required")
    return _np_convert(exif_analyzer.analyze(path))


def handle_detect_objects(params: Dict) -> Dict:
    image = _load_image(params)
    return _np_convert(object_detector.detect_all(image))


def handle_decompose_layers(params: Dict) -> Dict:
    image = _load_image(params)
    layers = layer_analyzer.decompose(image, num_layers=params.get("num_layers", 5))
    return {
        "num_layers": len(layers),
        "layers_base64": [_image_to_base64(l) for l in layers],
    }


def handle_enhance_edges(params: Dict) -> Dict:
    image = _load_image(params)
    enhanced = edge_enhancer.enhance_facial_edges(image, strength=params.get("strength", 1.5))
    return {"success": True, "image_base64": _image_to_base64(enhanced)}


def handle_zoom_enhance(params: Dict) -> Dict:
    image = _load_image(params)
    region = (params["x"], params["y"], params["width"], params["height"])
    enhanced = edge_enhancer.zoom_and_enhance(image, region, scale=params.get("scale", 2.0))
    return {"success": True, "image_base64": _image_to_base64(enhanced)}


def handle_full_analysis(params: Dict) -> Dict:
    path = params.get("image_path")
    if not path:
        raise ValueError("image_path required")

    job_id = str(uuid.uuid4())[:8]
    streaming = params.get("stream", False)

    if streaming:
        stream_emitter.register(job_id)
        stream_emitter.emit(job_id, "started", {"job_id": job_id})

    # Run analysis steps with progress
    if streaming:
        stream_emitter.emit(job_id, "progress", {"step": "exif_analysis", "pct": 10})
    result = forensic_analyzer.full_analysis(path)
    result = _np_convert(result)

    if streaming:
        stream_emitter.emit(job_id, "progress", {"step": "complete", "pct": 100})
        stream_emitter.emit(job_id, "complete", result)

    result["job_id"] = job_id
    return result


def handle_list_models(params: Dict) -> Dict:
    return {"providers": model_adapter.list_providers()}


def handle_set_model(params: Dict) -> Dict:
    provider = params.get("provider", "openai")
    ok = model_adapter.set_provider(provider)
    return {"success": ok, "active_provider": model_adapter.active_provider}


TOOL_HANDLERS = {
    "enhance_image": handle_enhance_image,
    "detect_mask": handle_detect_mask,
    "reconstruct_face": handle_reconstruct_face,
    "analyze_exif": handle_analyze_exif,
    "detect_objects": handle_detect_objects,
    "decompose_layers": handle_decompose_layers,
    "enhance_edges": handle_enhance_edges,
    "zoom_enhance": handle_zoom_enhance,
    "full_analysis": handle_full_analysis,
    "list_models": handle_list_models,
    "set_model": handle_set_model,
}


# ---------------------------------------------------------------------------
# MCP JSON-RPC server (stdio transport)
# ---------------------------------------------------------------------------

class MCPServer:
    """Model Context Protocol server using JSON-RPC over stdio."""

    def __init__(self):
        self.name = "revvel-forensic-mcp"
        self.version = "2.0.0"

    def _build_response(self, id: Any, result: Any) -> Dict:
        return {"jsonrpc": "2.0", "id": id, "result": result}

    def _build_error(self, id: Any, code: int, message: str) -> Dict:
        return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}

    def handle_initialize(self, id: Any, params: Dict) -> Dict:
        return self._build_response(id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
                "streaming": True,
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
        })

    def handle_tools_list(self, id: Any, params: Dict) -> Dict:
        tools = []
        for name, spec in MCP_TOOLS.items():
            tools.append({
                "name": name,
                "description": spec["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": spec["parameters"],
                },
            })
        return self._build_response(id, {"tools": tools})

    def handle_tools_call(self, id: Any, params: Dict) -> Dict:
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in TOOL_HANDLERS:
            return self._build_error(id, -32601, f"Unknown tool: {tool_name}")

        try:
            result = TOOL_HANDLERS[tool_name](arguments)
            return self._build_response(id, {
                "content": [{"type": "text", "text": json.dumps(result, default=str)}],
                "isError": False,
            })
        except Exception as exc:
            return self._build_response(id, {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            })

    def handle_request(self, request: Dict) -> Optional[Dict]:
        method = request.get("method", "")
        id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return self.handle_initialize(id, params)
        elif method == "notifications/initialized":
            return None  # notification, no response
        elif method == "tools/list":
            return self.handle_tools_list(id, params)
        elif method == "tools/call":
            return self.handle_tools_call(id, params)
        elif method == "ping":
            return self._build_response(id, {})
        else:
            if id is not None:
                return self._build_error(id, -32601, f"Method not found: {method}")
            return None

    def run_stdio(self):
        """Main loop — read JSON-RPC from stdin, write to stdout."""
        logger.info(f"{self.name} v{self.version} starting on stdio")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                sys.stdout.write(json.dumps(
                    self._build_error(None, -32700, "Parse error")
                ) + "\n")
                sys.stdout.flush()
                continue

            response = self.handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response, default=str) + "\n")
                sys.stdout.flush()


# ---------------------------------------------------------------------------
# Blue Ocean: Agent integration helpers
# ---------------------------------------------------------------------------

class AgentIntegration:
    """Helpers for integrating with popular AI agent frameworks."""

    @staticmethod
    def langchain_tools() -> List[Dict]:
        """Return tool definitions compatible with LangChain's StructuredTool format."""
        tools = []
        for name, spec in MCP_TOOLS.items():
            tools.append({
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
                "handler": f"revvel_forensic_mcp.TOOL_HANDLERS['{name}']",
            })
        return tools

    @staticmethod
    def openai_function_defs() -> List[Dict]:
        """Return tool definitions as OpenAI function-calling format."""
        functions = []
        for name, spec in MCP_TOOLS.items():
            functions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec["description"],
                    "parameters": {
                        "type": "object",
                        "properties": spec["parameters"],
                    },
                },
            })
        return functions

    @staticmethod
    def crewai_tools() -> List[Dict]:
        """Return tool definitions compatible with CrewAI."""
        return [
            {"name": name, "description": spec["description"], "args_schema": spec["parameters"]}
            for name, spec in MCP_TOOLS.items()
        ]

    @staticmethod
    def autogen_tools() -> List[Dict]:
        """Return tool definitions compatible with AutoGen."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec["description"],
                    "parameters": {"type": "object", "properties": spec["parameters"]},
                },
            }
            for name, spec in MCP_TOOLS.items()
        ]


agent_integration = AgentIntegration()


# ---------------------------------------------------------------------------
# HTTP transport (optional, for SSE streaming)
# ---------------------------------------------------------------------------

def create_http_app():
    """Create a FastAPI app for HTTP/SSE transport (Blue Ocean)."""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse, JSONResponse

    http_app = FastAPI(
        title="Revvel Forensic MCP Server (HTTP)",
        version="2.0.0",
    )

    @http_app.get("/")
    async def root():
        return {"name": "revvel-forensic-mcp", "version": "2.0.0", "transport": "http"}

    @http_app.get("/tools")
    async def list_tools():
        return {"tools": list(MCP_TOOLS.keys())}

    @http_app.post("/tools/{tool_name}")
    async def call_tool(tool_name: str, params: Dict = {}):
        if tool_name not in TOOL_HANDLERS:
            return JSONResponse(status_code=404, content={"error": f"Unknown tool: {tool_name}"})
        try:
            result = TOOL_HANDLERS[tool_name](params)
            return JSONResponse(content=result)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @http_app.get("/stream/{job_id}")
    async def stream_events(job_id: str):
        """SSE endpoint for streaming job progress (Blue Ocean)."""
        return StreamingResponse(
            stream_emitter.stream(job_id),
            media_type="text/event-stream",
        )

    @http_app.get("/integrations/langchain")
    async def langchain_defs():
        return {"tools": agent_integration.langchain_tools()}

    @http_app.get("/integrations/openai")
    async def openai_defs():
        return {"functions": agent_integration.openai_function_defs()}

    @http_app.get("/integrations/crewai")
    async def crewai_defs():
        return {"tools": agent_integration.crewai_tools()}

    @http_app.get("/integrations/autogen")
    async def autogen_defs():
        return {"tools": agent_integration.autogen_tools()}

    @http_app.get("/models")
    async def list_models():
        return {"providers": model_adapter.list_providers()}

    @http_app.post("/models/{provider}")
    async def set_model(provider: str):
        ok = model_adapter.set_provider(provider)
        return {"success": ok, "active": model_adapter.active_provider}

    return http_app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")

    if transport == "http":
        import uvicorn
        http_app = create_http_app()
        uvicorn.run(http_app, host="0.0.0.0", port=int(os.getenv("MCP_PORT", "9000")))
    else:
        server = MCPServer()
        server.run_stdio()
