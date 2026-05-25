# Changelog

All notable changes to this project are documented in this file.

## [2.0.2] - 2026-05-25

### Added
- Expanded `/api/static/index.html` into a full S2M website surface with public Vercel link, pricing, research, and artifact coverage.

### Changed
- Fixed website navigation links to use the live FastAPI routes (`/docs`, `/redoc`, `/health`).
- Updated README, deployment, and go-to-market docs to trace the website source and public link.
- Tightened baseline validation scripts to verify the website surface and deployment traceability automatically.

## [2.0.1] - 2026-05-23

### Added
- Revvel-standards baseline documentation set:
  - `DEPLOYMENT_GUIDE.md`
  - `GO_TO_MARKET.md`
  - `BRAND_GUIDELINES.md`
  - `SECURITY.md`
  - `REVVEL_STANDARDS_S2M.md`
- Baseline repository validation scripts:
  - `scripts/test-baseline.js`
  - `scripts/build-baseline.js`
- Root `package.json` with `npm test` and `npm run build` commands.

### Changed
- Updated root `README.md` with revvel-standards traceability and baseline validation usage.
