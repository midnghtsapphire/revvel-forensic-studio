# Revvel Forensic MCP Server

Standalone Model Context Protocol (MCP) server for AI agent integration with forensic image analysis capabilities. Split from the `revvel-forensic-studio` monorepo and enhanced with Blue Ocean features.

## Features

### MCP Tools
- **enhance_image** — Apply beauty enhancements (natural, glamour, dramatic, subtle)
- **detect_mask** — Detect if a face is wearing a mask
- **reconstruct_face** — Reconstruct face from masked image
- **analyze_exif** — Extract and analyse EXIF metadata
- **detect_objects** — Detect glasses, jewelry, tags, tattoos
- **decompose_layers** — Frequency-based layer decomposition
- **enhance_edges** — CLAHE + Canny edge enhancement
- **zoom_enhance** — Region zoom with detail recovery
- **full_analysis** — Comprehensive forensic analysis (with streaming)
- **list_models** — List available AI model providers
- **set_model** — Switch active AI model provider

### Blue Ocean Enhancements
- **Streaming Results** — SSE-based streaming for long-running operations via HTTP transport
- **Multi-Model Support** — Switch between OpenAI, Anthropic, Google, OpenRouter, and local models
- **AI Agent Integration** — Pre-built tool definitions for LangChain, CrewAI, AutoGen, and OpenAI function calling
- **Dual Transport** — stdio (standard MCP) and HTTP (for web-based agents)

## Quick Start

### stdio Transport (Standard MCP)

```bash
pip install -r requirements.txt
python server.py
```

Configure in your MCP client (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "revvel-forensic": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/revvel-forensic-mcp"
    }
  }
}
```

### HTTP Transport (Blue Ocean)

```bash
MCP_TRANSPORT=http MCP_PORT=9000 python server.py
```

Then access:
- Tool list: `GET /tools`
- Call tool: `POST /tools/{tool_name}`
- SSE stream: `GET /stream/{job_id}`
- Agent integrations: `GET /integrations/{framework}`

## Agent Framework Integration

### LangChain
```python
import requests
tools = requests.get("http://localhost:9000/integrations/langchain").json()
```

### OpenAI Function Calling
```python
functions = requests.get("http://localhost:9000/integrations/openai").json()
```

### CrewAI
```python
tools = requests.get("http://localhost:9000/integrations/crewai").json()
```

### AutoGen
```python
tools = requests.get("http://localhost:9000/integrations/autogen").json()
```

## Docker

```bash
# stdio transport
docker build -t revvel-forensic-mcp .
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | docker run -i revvel-forensic-mcp

# HTTP transport
docker run -p 9000:9000 -e MCP_TRANSPORT=http revvel-forensic-mcp
```

## Configuration

See `.env.example` for all environment variables.

| Variable | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | stdio | Transport mode: stdio or http |
| `MCP_PORT` | 9000 | HTTP port (when transport=http) |
| `MCP_MODEL_PROVIDER` | openai | Active AI model provider |
| `OPENAI_API_KEY` | | OpenAI API key |
| `ANTHROPIC_API_KEY` | | Anthropic API key |
| `OPENROUTER_API_KEY` | | OpenRouter API key |
| `FACE_BACKEND` | auto | Face detection backend |

## Architecture

```
revvel-forensic-mcp/
├── server.py               # MCP server (stdio + HTTP)
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
