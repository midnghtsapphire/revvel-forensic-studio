# Revvel Standards — S2M Trigger Policy

## S2M Trigger
If a work request, issue title, or PR title includes `S2M` (case-insensitive), the repository where that request is opened is treated as a full **Ship-to-Market** implementation target.

## Required S2M Scope
For S2M-tagged work, delivery must include:
1. Revvel-standards baseline docs and validation scripts.
2. End-to-end ship plan tied to this repository's actual surfaces.
3. Research engines, implementation suggestions, assets inventory, and artifact index.
4. One-iteration canonical implementation (no parallel alternative implementations).

## Research Engines (Minimum)
- Market intelligence and competitor monitoring.
- Standards/compliance signal monitoring.
- Product usage signal aggregation.
- Revenue and growth modeling.

## Suggestions (Minimum)
- Prioritized product improvements mapped to segment value.
- Distribution and launch tactics for technical and non-technical buyers.
- Monetization and conversion recommendations.

## Assets and Artifacts (Minimum)
- Deployable assets inventory (API, CLI, MCP, website-in-test).
- Documentation artifacts index.
- Validation artifact outputs (`npm test`, `npm run build`).

## Repository Mapping for This Project
- Repository: `midnghtsapphire/revvel-forensic-studio`
- Ship-to-market target: forensic studio API + CLI + MCP + website-in-test path.
