# Security Policy

## Supported Components
Security fixes are applied to active components in this repository:
- `api/`
- `cli/`
- `mcp/`

## Reporting a Vulnerability
Please open a private security report through GitHub Security Advisories for this repository.

Include:
- Affected component and version/commit
- Reproduction steps
- Impact assessment
- Suggested remediation (if available)

## Security Baseline
- Do not commit secrets (`.env` files are ignored).
- Use environment variables for API provider keys and webhook credentials.
- Validate and sanitize all inbound API parameters.
- Keep dependencies patched and re-run baseline checks before releases.
