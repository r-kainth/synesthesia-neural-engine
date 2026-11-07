# Synesthesia Engine: Design Doc

Oct 19, 2026 · @ritik

## Overview

Synesthesia Engine is a live, in-browser music visualizer whose visuals are driven by ML models trained from scratch in PyTorch, and which learns from its own users which visuals fit which sounds. It replaces the v0 plan (Whisper + librosa + Gemini + Stable Diffusion), which trained nothing, could not run in real time, and needed a costly always-on GPU to deploy.

### What makes it original

- **Procedural, not generative imagery.** Visuals are GPU shaders with a small set of named parameters, rendered at 60 fps. ML decides the parameters; it never paints pixels.
- **Grounded in synesthesia research.** The starting sound-to-sight mapping encodes documented cross-modal correspondences (pitch to lightness, loudness to size, timbre sharpness to angularity).
- **It learns from people.** The public site asks visitors which of two variants fits the music better. Those votes train a reward model, and the mapping is improved against it: a small, honest version of the RLHF idea.

### Goals

1. Real hands-on depth in PyTorch: datasets, training loops, evaluation, export. Written by hand, understood line by line.
2. A public URL on AWS, built with infrastructure as code and CI/CD, that costs close to nothing to run.
3. A portfolio piece with evidence: baselines, metrics, ablations, decision records, and a model card.

### Non-goals (for now)

- Generating images or video with diffusion models.
- Accounts, logins, or social features.
- Mobile-native apps. The browser is the platform.
- Storing anyone's audio, ever.

## Product experience

A visitor opens the site, picks a sound source, and the visuals react instantly; no signup, no upload to a server.

1. **Safety notice first.** A short photosensitivity warning with a "reduce flashing" toggle on by default. Fast strobing can trigger seizures; the reduced mode caps hard flashes at well under 3 per second (the WCAG threshold).
2. **Pick a source.** Microphone (live room audio), a local audio file (decoded in the browser), or audio from another browser tab (for example Spotify's web player) via screen-capture audio, best supported in Chrome and Edge. The site never connects to Spotify or any streaming service directly; it only hears the tab the visitor chooses to share.
3. **Instant reaction.** Within one frame, beats, loudness and brightness drive the visuals. This is the reflex layer, with no ML in the loop.
4. **The mood settles in.** After about 3 seconds of audio, the perception model starts reporting mood (calm/energetic, dark/bright), and the palette, motion and texture drift toward it, smoothly.
5. **Occasional question.** Every few minutes, a small prompt: "Which fits better, A or B?" The visuals switch between two variants for a few seconds each and the visitor taps one. Skippable, never blocking.
6. **Controls.** Pick a visual "world" (shader), adjust sensitivity, go fullscreen.

### Latency budget

| Layer | Update rate | Max delay | Why it matters |
| --- | --- | --- | --- |
| Reflex (DSP) | every audio frame, about 10 to 20 ms | under 30 ms | A beat must land on the beat |
| Perception model | every 250 to 500 ms, on a 3 s window | under 20 ms per inference | Mood changes slowly; it can lag a little |
| Renderer | 60 fps (16.7 ms per frame) | 1 frame | Smooth interpolation hides model updates |

## System architecture

Four layers run on the visitor's machine; AWS serves the site and model files and collects votes, which feed retraining.

&#91;embedded content: System architecture · 4 browser layers, 3 cloud pieces\]

Read it top to bottom: sound goes in, two layers listen (one fast and fixed, one slow and learned), the mapping network turns what they hear into shader parameters, and the renderer draws. The loop on the right is how the system improves: votes are stored, a training job builds a better mapping, and it is published back through S3 only if it beats the current one. Both browser models load from S3 by version, using the manifest.

## ML design

Three models, built in order: a perception model that hears mood, a mapping network that turns what it hears into visual parameters, and a reward model that learns from votes which mappings people prefer.

### 1. Perception model (M1)

**Task.** Given about 3 seconds of audio as a log-mel spectrogram, predict valence (negative to positive feeling) and arousal (calm to energetic), plus a small embedding (around 32 numbers) summarizing the sound's character.

**Data.**

- **DEAM** (MediaEval Database for Emotional Analysis in Music): roughly 1,800 tracks with valence/arousal annotated every half second. Time-varying labels are exactly what a live visualizer needs. Licence terms to be checked before use.
- **MTG-Jamendo**: roughly 55k Creative Commons tracks with mood/theme tags. Optional pretraining on a multi-label mood task, then fine-tune on DEAM.

**Input pipeline.** Mono audio, resampled to 16 kHz (or 22.05 kHz), STFT, 64 mel bands, log scale, normalized with training-set statistics. Augmentation: random gain, small pitch shift, time and frequency masking (SpecAugment).

**Model ladder.** Each rung must beat the one before, or it isn't worth its complexity.

| Rung | Model | Purpose |
| --- | --- | --- |
| 0 | Predict the training mean | The floor. Anything worse is broken. |
| 1 | Ridge regression on hand-made features (librosa) | Shows how far classic features get |
| 2 | Small CNN on mel spectrograms | First real deep model |
| 3 | CRNN (CNN + GRU over time) | Captures how sound evolves within the window |
| 4 | Small audio transformer (optional) | Only if rung 3 plateaus |

**Size limit.** The final model must fit the browser: under about 5 MB and under 20 ms per inference on a laptop CPU.

**Evaluation.** Concordance correlation coefficient (CCC), the standard metric for this task, plus RMSE. CCC rewards both correlation and getting the scale right:

```latex
\mathrm{CCC} = \frac{2\rho\sigma_x\sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}
```

**Split by song, never by window.** Windows from one song look alike; mixing them across train and test inflates scores. All splits group by track ID.

### 2. Mapping network (M2, then M4)

**Task.** Turn a context vector c (DSP summary + valence/arousal + embedding) into a visual parameter vector θ, each value in \[0, 1\] (palette, warp, particle density, symmetry, speed, sharpness, and so on). The parameter list lives in one shared schema file read by both Python and TypeScript.

**Version 0, the research prior.** Hand-written rules from cross-modal correspondence research: higher spectral centroid gives lighter colours, louder gives bigger and brighter, harsher timbre gives sharper shapes, high arousal gives faster motion, positive valence gives warmer hues.

**Version 1, learned.** A small MLP first trained to imitate the prior (so it starts sensible), then improved using the reward model below, while a penalty keeps it from drifting far from the prior:

```latex
\max_f \; \mathbb{E}_c\left[ r(c, f(c)) \right] - \lambda \, \mathbb{E}_c \left\| f(c) - \mathrm{prior}(c) \right\|^2
```

That penalty guards against "reward hacking": finding odd parameter settings the reward model scores highly but people would not.

### 3. Reward model from votes (M4)

**Data.** Each vote stores the context c, the two parameter vectors shown (θ\_A, θ\_B) and the choice. Variant B is the current mapping plus random noise, which doubles as exploration.

**Model.** A small network r(c, θ) scoring how well visuals fit sound, trained with the Bradley–Terry model, the same pairwise idea used in RLHF:

```latex
P(A \succ B \mid c) = \sigma\big(r(c, \theta_A) - r(c, \theta_B)\big)
```

**Evaluation.** Accuracy on held-out votes (chance = 50%). The final test is a blind A/B on the live site: does the learned mapping beat the research prior?

**Cold start.** Hundreds to low thousands of votes will be needed (a rough estimate). Seed with friends and classmates, and keep the prior as a fallback until the numbers say otherwise.

## Tech stack

Python for everything ML and the API, TypeScript for everything in the browser, Terraform for the cloud. Each choice favours learning the fundamentals over hiding them.

| Area | Choice | Why this one |
| --- | --- | --- |
| Python tooling | Python 3.12, uv, ruff, pytest | uv makes environments fast and reproducible; ruff and pytest keep code honest |
| Deep learning | PyTorch + torchaudio, plain training loops | Writing the loop yourself is the point. Lightning can come later, once you know what it saves you |
| Classic audio features | librosa (offline only) | Baselines and sanity checks against the browser's features |
| Experiment tracking | Weights & Biases (free personal tier) | Every run's config, curves and metrics, comparable side by side |
| Configs | YAML files + Python dataclasses | Every experiment reproducible from one file |
| Model export | ONNX + onnxruntime (Python) for checks | One portable model file for the browser; Python runtime used to verify it |
| Browser inference | onnxruntime-web (WASM, WebGPU when available) | Runs the model on the visitor's machine, so no GPU server is needed |
| Audio in the browser | Web Audio API + AudioWorklet | Low-latency feature extraction off the main thread |
| Rendering | WebGL2 fragment shaders (thin helper such as twgl.js); WebGPU later | WebGL2 works almost everywhere; raw shaders teach GLSL properly |
| Frontend build | TypeScript + Vite, Vitest | Fast dev server, typed code, tests |
| API | FastAPI on AWS Lambda | Same language as the ML code; tiny, serverless |
| Infrastructure | Terraform | The whole cloud setup in code, reviewable and repeatable |
| CI/CD | GitHub Actions, AWS access via OIDC | No long-lived AWS keys stored in GitHub |

**Hardware note.** Small models train fine on a laptop CPU. For heavier runs use a free Colab or Kaggle GPU. PyTorch GPU support on Windows with an AMD card is limited, so do not plan around it.

**Train/serve parity.** The model's input features must be computed identically in Python (training) and TypeScript (browser). Two options: compute the mel spectrogram inside the ONNX graph so the browser sends raw audio, or keep golden test clips with expected features and test both sides against them. Either way, a parity test runs in CI.

## Cloud architecture (AWS)

Because all inference runs in the visitor's browser, the cloud only serves files and stores votes, so the whole setup is serverless and should cost a few dollars a month or less.

| Piece | AWS service | Job |
| --- | --- | --- |
| Website | S3 (private bucket) + CloudFront | Serves the built frontend over HTTPS from a global cache |
| Model files | Same S3 bucket, versioned paths (`/models/perception/v3/model.onnx`) + a `manifest.json` | The site reads the manifest to know which model versions to load; rollback = edit one file |
| Votes API | API Gateway (HTTP API) + Lambda running FastAPI | `POST /votes`, `GET /manifest` |
| Vote storage | DynamoDB (on-demand) | Pay per request; nothing to keep running |
| Retraining | Early: run locally or on Colab, upload through CI. Later: scheduled job (EventBridge + SageMaker training job or AWS Batch on spot GPU) | Pull votes, train, evaluate, publish only if the new model wins |
| Monitoring | CloudWatch logs and alarms | Errors and request counts |
| Domain (optional) | Route 53 + ACM certificate | A nice URL |

### Cost guardrails (set these up before anything else)

- AWS Budgets alerts at small thresholds (for example $5 and $15).
- No NAT gateways and no always-on EC2 instances; these are the classic surprise bills.
- CloudWatch log retention set to 14 days.
- API Gateway throttling on the votes endpoint.
- Check the current free-tier terms when creating the account; they have changed in the past.

### Privacy

- Audio never leaves the browser. Not uploaded, not logged, not stored.
- A vote stores only: derived feature summary, the two parameter vectors, the choice, model versions, and a random session ID. No IP addresses, no accounts. Each vote also records its audio source (mic, file or tab); votes from tab-captured sessions are never used for training, because streaming services' terms restrict feeding their content into ML models.
- A plain-language privacy note on the site says exactly this.

### Environments

`dev` and `prod`, each its own Terraform state. Merges to `main` deploy to dev automatically; prod deploys from a tagged release.

## Repo structure

One monorepo with four top-level parts (ml, web, api, infra), plus a `shared/` folder holding the contracts both languages must agree on.

```
synesthesia-engine/
├── CLAUDE.md                      # standing instructions for Claude Code
├── README.md                      # demo GIF, what it is, how to run
├── docs/
│   ├── design.md                  # snapshot of this doc, exported at each milestone
│   ├── decisions/                 # ADR-0001-live-visualizer.md, ...
│   └── notes/                     # your learning notes, in your words
├── shared/
│   ├── visual_params.schema.json  # single source of truth for θ (names, ranges, defaults)
│   └── golden/                    # short test clips + expected features (parity tests)
├── ml/
│   ├── pyproject.toml
│   ├── configs/                   # one YAML per experiment
│   ├── src/synesthesia_ml/
│   │   ├── data/                  # download.py, deam.py (Dataset), splits.py
│   │   ├── features/              # mel.py, handcrafted.py
│   │   ├── models/                # baselines.py, cnn.py, crnn.py, mapping.py, reward.py
│   │   ├── train/                 # loop.py, metrics.py (CCC), callbacks.py
│   │   └── export/                # to_onnx.py, parity.py
│   ├── notebooks/                 # exploration only; nothing imported from here
│   └── tests/
├── web/
│   ├── package.json, vite.config.ts
│   ├── src/
│   │   ├── audio/                 # input.ts (mic/file/tab), feature-worklet.ts, features.ts
│   │   ├── inference/             # session.ts (onnxruntime-web), scheduler.ts
│   │   ├── mapping/               # prior.ts (research rules), params.ts (schema loader)
│   │   ├── render/                # renderer.ts, worlds/*.frag
│   │   ├── feedback/              # compare.ts (A/B prompt), api.ts
│   │   └── main.ts
│   └── tests/
├── api/
│   ├── pyproject.toml
│   ├── src/synesthesia_api/       # app.py (FastAPI), schemas.py, storage.py (DynamoDB)
│   └── tests/
├── infra/
│   ├── modules/                   # site, api, data
│   └── envs/                      # dev/, prod/
└── .github/workflows/             # ci.yml, deploy-web.yml, deploy-api.yml, deploy-infra.yml
```

**Rules that keep it clean**

- Notebooks explore; real code lives in `src/` and has tests.
- Anything both Python and TypeScript use (parameter names, feature definitions) lives in `shared/` and nowhere else.
- Every non-obvious choice gets a short decision record in `docs/decisions/`.

## Milestones

Five milestones take the project from a beat-reactive shader (late November) to a public site (early March) to a model that learns from its visitors (late April). The dates are a target, not a promise (reset on Nov 7 after a three-week pause): they assume about 8 to 10 hours a week and slip around exams.

&#91;embedded content: Roadmap · 6 phases, each with exit criteria\]

Each phase ends with a gate: don't start the next milestone until the exit criteria are met. M1 is deliberately the longest, because it is where most of the learning happens, and it spans December exams and the winter break: expect a slow stretch, then a fast one.

## Week 1 (from Nov 7): reset and first light

By the end of Day 7 (target: Sun Nov 15), you should have a clean repo and a shader in the browser that pulses with your microphone. That is the first piece of M0. Each day is about 1 to 1.5 hours of work. Days are numbered, not dated, so a busy week shifts the plan instead of breaking it.

### Day 1 (Sat Nov 7): skeleton and first Claude Code session

- [x] Tag the old code `archive/v0` and push a branch `v0-archive`, so nothing is lost
- [x] Clear `main` with `git rm -r .`, commit, and push
- [x] Decide what's committed: `CLAUDE.md` and `docs/design.md` committed, `CLAUDE.local.md` gitignored
- [ ] Add `README.md` (one paragraph), `LICENSE` (MIT), `.gitignore` (Python + Node, plus `CLAUDE.local.md`), and the folder skeleton above
- [ ] Put `CLAUDE.md`, `CLAUDE.local.md` and `docs/design.md` in place
- [ ] Open the repo in Claude Code and ask it to explain the skeleton back to you

### Day 2: Python environment

- [ ] Install uv; create `ml/pyproject.toml` with torch, torchaudio, librosa, numpy, matplotlib, pytest, ruff
- [ ] Run a 5-line PyTorch check: create a tensor, multiply, print the device
- [ ] Write `docs/notes/00-setup.md`: what uv, a virtual environment and a lockfile are, in your words

### Day 3: hear like a computer (notebook)

- [ ] Load one song with librosa; plot the waveform, STFT, mel spectrogram, chroma and onset strength
- [ ] Run beat tracking and check the beats against your ears
- [ ] Write `docs/notes/01-audio-features.md`: one paragraph per feature, what it measures and what it might drive visually

### Day 4: browser scaffold

- [ ] Create `web/` with Vite + TypeScript
- [ ] Get microphone audio with `getUserMedia`, read loudness with an `AnalyserNode`
- [ ] Draw a fullscreen WebGL2 quad with a fragment shader whose brightness follows loudness

### Day 5: the reflex layer

- [ ] Compute band energies (bass, mids, highs) and spectral flux onset detection in TypeScript
- [ ] Smooth each signal with an attack/release envelope so motion is punchy but not jittery
- [ ] Make a beat pulse; compare your onset times against librosa's on the same clip (your first parity check)

### Day 6: parameters and the first research rules

- [ ] Write `shared/visual_params.schema.json` v0 with about 8 parameters (name, range, default)
- [ ] Wire them to shader uniforms, and hand-code three rules: spectral centroid to brightness, loudness to scale, onset to pulse
- [ ] Add the photosensitivity notice and the reduce-flashing toggle

### Day 7: read, decide, show it off

- [ ] Skim the DEAM and MTG-Jamendo pages and papers; confirm licence terms
- [ ] Write ADR-0001 (live visualizer, not offline render), ADR-0002 (inference in the browser) and ADR-0003 (no Spotify API; tab capture instead)
- [ ] Quick spike: share a Spotify web player tab with `getDisplayMedia` in Chrome and confirm the audio reaches your visualizer
- [ ] Record a 20-second screen capture for the README; note what felt good and what felt off

**Done means:** a fresh clone runs `uv sync` and `npm run dev`, and the visuals visibly react to music from your mic.

## Learning plan, risks and decisions

Learn each concept just before the milestone that needs it, and write it down in your own words in `docs/notes/`; if you can't explain it, you don't own it yet.

### What to learn, and when

| Milestone | Concepts | Good starting resources |
| --- | --- | --- |
| M0 | Sampling, FFT/STFT, spectrograms, onsets; GLSL basics | Valerio Velardo's "Audio Signal Processing for ML" videos; The Book of Shaders |
| M1 | Tensors, autograd, Dataset/DataLoader, training loop, overfitting, CNNs, RNNs | Official PyTorch "Learn the Basics" and torchaudio tutorials; Karpathy's "Neural Networks: Zero to Hero" |
| M2 | ONNX export, quantization, WebAssembly, browser threading | ONNX Runtime web docs |
| M3 | IAM, S3/CloudFront, Lambda, DynamoDB, Terraform, OIDC | AWS and Terraform official tutorials |
| M4 | Pairwise preference learning, Bradley–Terry, reward hacking, A/B testing | RLHF overview papers and blog posts (reward-modelling section) |

### How to work with Claude Code without it becoming slop

- **You write the ML core** (Dataset, model, loop, metrics). Claude Code explains, reviews and suggests; you type.
- **Claude Code may write the scaffolding** (configs, Terraform boilerplate, CI YAML, CSS), but you read every diff before accepting.
- **Ask "why", not just "do"**: "Explain why the loss plateaus" teaches more than "fix the loss".
- **One task per session**, starting from this doc and `CLAUDE.md`, ending with a commit.

### Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Mood labels are noisy and subjective | Model accuracy plateaus low | Honest baselines; report CCC alongside the literature, not against perfection |
| Python and browser features differ | Model behaves differently live than in testing | Golden-file parity tests in CI, or mel spectrogram inside the ONNX graph |
| Slow devices | Dropped frames | Small model, inference off the main thread, graceful fallback to reflex-only |
| Too few votes | Reward model can't learn | Research prior stays the default; seed votes from friends; keep the A/B prompt light |
| Scope creep | Never ships | Each milestone has exit criteria; stretch goals wait for M5 |
| Surprise AWS bill | Money | Budgets alerts, serverless only, no NAT gateways |
| Dataset licences | Can't publish models | Check licences in week 1; prefer Creative Commons data |

### Open questions

- [ ] Which visual "world" first: flowing fields, particles, or geometric kaleidoscope?
- [ ] Realistic hours per week through the winter term?
- [ ] Does tab capture work reliably with Spotify's web player in Chrome and Edge? (answered by the week 1 spike)

### Decision log

| Date | Decision | Why |
| --- | --- | --- |
| Nov 7, 2026 | This live doc is the source of truth for Claude Code too; `docs/design.md` is a snapshot exported at each milestone | Claude Code can read (and update) the live doc directly, so no constant re-exporting; the repo still gets a public copy |
| Nov 7, 2026 | Split instructions: `CLAUDE.md` (project rules, committed) and `CLAUDE.local.md` (personal and learning-mode rules, gitignored); `docs/design.md` committed | Public repo shows project conventions and design evidence; personal working notes stay private; Claude Code loads both |
| Nov 7, 2026 | Schedule reset to start Nov 7; week 1 days numbered, not dated | Three weeks slipped; numbered days don't go stale when a week gets busy |
| Oct 19, 2026 | No Spotify API; tab audio capture instead, moved into M2 | The API never provides audio; audio-analysis endpoints are closed to new apps; Spotify's Developer Policy bars feeding its content into ML models; Development Mode allows only a few allowlisted users |
| Oct 19, 2026 | Restart from scratch; archive the v0 stitch | v0 trained nothing and could not run live |
| Oct 19, 2026 | Live visualizer, not offline render | Real-time feel is the point of synesthesia |
| Oct 19, 2026 | Procedural shaders, not diffusion | 60 fps, no GPU server, full control |
| Oct 19, 2026 | Inference in the browser via ONNX | Near-zero hosting cost; no audio leaves the device |
| Oct 19, 2026 | AWS over GCP | Complements Azure experience from the OPS co-op |
