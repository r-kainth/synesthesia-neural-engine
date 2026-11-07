# Synesthesia Engine

A live, in-browser music visualizer. Procedural shaders render at 60 fps; small
PyTorch models (exported to ONNX, run in the browser) decide the shader
parameters; votes from visitors train a reward model that improves the mapping.

The design is described in `docs/design.md`, a snapshot exported at the end of
each milestone (the maintainer's working copy may be newer; see
`CLAUDE.local.md` if present). Read the relevant section before starting any
milestone task. Decisions are recorded in `docs/decisions/`.

## Working style

- Small, reviewable diffs. One task per session, ending with a commit.
- Explain non-obvious changes in the commit message or PR description.
- If something in the design doc looks wrong or outdated, say so rather than
  silently working around it.
- Personal instructions, if present, live in `CLAUDE.local.md` (gitignored).

## Layout

- `ml/` Python: data, features, models, training, ONNX export (uv project)
- `web/` TypeScript + Vite: audio input, AudioWorklet features, onnxruntime-web,
  mapping, WebGL2 renderer, A/B feedback UI
- `api/` Python FastAPI on AWS Lambda: votes and model manifest
- `infra/` Terraform: `modules/` and `envs/dev`, `envs/prod`
- `shared/` contracts used by BOTH Python and TypeScript:
  - `visual_params.schema.json` is the single source of truth for shader
    parameter names, ranges, defaults. Never hard-code parameter lists elsewhere.
  - `golden/` test clips + expected features for parity tests
- `docs/notes/` learning notes; `docs/decisions/` ADRs

## Commands

- ML: `cd ml && uv sync`, tests `uv run pytest`, lint `uv run ruff check .`
- Web: `cd web && npm install`, dev `npm run dev`, tests `npm test`
- API: `cd api && uv sync`, tests `uv run pytest`
- Infra: `cd infra/envs/dev && terraform plan` (never `apply` without approval)

## Rules

- Notebooks (`ml/notebooks/`) are for exploration only; nothing imports from them.
- Data splits group by track ID. Never split windows of one song across
  train/val/test.
- Every model change is compared against the previous best and the baselines
  (mean predictor, ridge regression) on held-out songs, using CCC.
- Feature computation must match between Python and TypeScript. If you touch
  feature code on either side, run or update the parity tests.
- Never store or upload user audio. Votes store only feature summaries,
  parameter vectors, the choice, the audio source (mic/file/tab), model
  versions, and a random session ID.
- No Spotify (or other streaming service) API integration. Streaming audio
  only arrives via tab capture, and votes from tab sessions are never used
  for training. See ADR-0003.
- Visuals: respect the reduce-flashing setting (hard flashes well under 3/s).
- AWS: no NAT gateways, no always-on instances, no long-lived keys
  (GitHub Actions uses OIDC). Ask before creating anything that costs money.
- Python: type hints, ruff-clean, pytest for anything in `src/`.
- TypeScript: strict mode, no `any` without a comment explaining why.
- New non-obvious decisions get a short ADR in `docs/decisions/`.

## Current status

- Milestone: M0 (reflex visualizer), restarted Nov 7, 2026. Old code is
  preserved at tag `archive/v0` and branch `v0-archive`; `main` was cleared (commit message accidently mentions tag is v0-archive, just ignore this).
- Update this line when a milestone's exit criteria are met.
