# Deployment Guide

This repository ships three deployable surfaces:
- API (`/api`)
- CLI (`/cli`)
- MCP server (`/mcp`)

## 1) Website in Test (Vercel)

Target URL: `https://revvel-forensic-studio.vercel.app`

Recommended structure for Vercel test surface:
- Host a lightweight frontend/static interface that calls deployed API endpoints.
- Store API keys and webhooks as Vercel environment variables.

## 2) API Deployment (Container)

1. Build image from `/api/Dockerfile`.
2. Set environment variables for API key management and webhooks.
3. Expose port `8000` and route HTTPS traffic.
4. Run health checks against the `/` or `/docs` endpoint.

## 3) MCP Server Deployment (Container/VM)

1. Build image from `/mcp/Dockerfile`.
2. Configure provider keys (OpenAI/Anthropic) via environment variables.
3. Expose server transport (stdio/http proxy) based on MCP client runtime.
4. Add process supervision and restart policy.

## 4) CLI Distribution

- Build Python package or binary wrappers from `/cli`.
- Publish with release notes tied to `CHANGELOG.md`.

## 5) CI Validation Baseline

Run before release:

```bash
npm test
npm run build
```

These commands validate revvel-standards baseline files and deployment-critical repository structure.
