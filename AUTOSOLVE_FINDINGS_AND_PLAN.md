# AutoSolve Findings and Plan

## Executive Summary

AutoSolve has a strong and useful aim: keep Blender users inside the open-source ecosystem for common camera tracking and scene setup workflows. The project already has a credible Blender-native pipeline for feature detection, bidirectional tracking, cleanup, solve refinement, region controls, smoothing, and scene setup.

The main gap is not ambition. The main gap is that the public product promise currently runs ahead of the proof. The add-on presents itself as adaptive, research-driven, and close to one-click industry-level solving, but the current public implementation is mostly deterministic Blender operator orchestration, heuristics, local statistical feedback, and manual validation.

The recommended direction is:

- Remove all public training-data collection and contribution surfaces from the Blender add-on.
- Reposition the add-on as an honest beta for assisted Blender-native tracking.
- Move neural-network work into offline developer tooling under `ml/`.
- Train models to optimize Blender-native tracking decisions, not to replace Blender's solver.
- Build a benchmark/evaluation loop before making industry-comparison claims.

## Current Product Aim

The strongest version of AutoSolve is:

> A Blender-native camera tracking assistant that automates the first pass, improves marker placement and cleanup, diagnoses common failures, and helps users get to a usable solve faster without leaving Blender.

This is a good target because it solves a real workflow problem:

- Beginners struggle with Blender's manual tracking workflow.
- Students and small creators often leave Blender for paid or cracked tools.
- Blender's native tracker is capable, but the surrounding workflow is technical and unforgiving.
- An add-on can add workflow intelligence without shipping an external solver.

The product should compete by making Blender's existing tracking tools easier, more consistent, and better configured.

## What Is Working Well

- The mission is clear and valuable: open-source, Blender-first camera tracking.
- The add-on uses Blender's native tracking system, which keeps installation simple.
- The UI is phase-based: track, setup scene, refine.
- The pipeline includes useful real-world tracking steps:
  - smart feature detection
  - bidirectional tracking
  - cleanup of short/jittery/high-error tracks
  - retry and refinement logic
  - optional smoothing
  - annotation-guided include/exclude regions
- The code already contains thoughtful domain heuristics:
  - footage type presets
  - robust mode
  - tripod mode
  - region coverage
  - weak-zone replenishment
  - high-error filtering

These are meaningful foundations. The addon does not need a full solver replacement to become useful.

## Key Findings and Gaps

### 1. Public claims are ahead of validation

The README currently frames AutoSolve around one-click solves that can rival industry giants. That is a powerful long-term goal, but it is not yet supported by a benchmark corpus or automated validation suite.

Missing proof:

- no public benchmark footage set
- no repeatable solve quality metrics
- no comparison baseline against Blender manual tracking or commercial tools
- no automated Blender smoke tests
- no documented success/failure rate by footage type

Recommendation:

- Treat "industry-level" as a long-term ambition, not a current product claim.
- Publish measured performance only after a benchmark harness exists.

### 2. AI and learning language is too inflated

The docs correctly acknowledge that the current system is not a deep neural network. However, the public-facing wording still leans on AI, adaptive learning, community data, and future model claims.

Current reality:

- The public add-on mainly uses heuristics and statistical local feedback.
- The model files and session recorders are not a true shipped neural-network tracking model.
- The learning/data-collection story creates trust and privacy risk.

Recommendation:

- Public docs should say "heuristic presets" and "Blender-native retry logic."
- Neural-network work should live in offline developer tooling until it is validated.

### 3. Public data collection should be removed

The public add-on currently exposes training-data contribution, export/import, research beta UI, local behavior recording, contributor IDs, session JSON, and model updates.

This conflicts with the desired public direction:

- no telemetry UI
- no automatic session or behavior recording
- no contributor IDs
- no public HuggingFace upload flow
- no local writes during ordinary solves for data collection

Recommendation:

- Remove public data collection entirely from the add-on.
- Leave old local data untouched but unused.
- Document that legacy beta data is no longer collected or read by the public add-on.

### 4. Runtime behavior and docs disagree

One concrete mismatch: public docs mention Fast/Balanced/Quality marker counts of 20/35/50, while runtime targets are higher because the implementation tracks more and then filters/averages.

Recommendation:

- Avoid exact marker counts in public docs unless they are generated from constants.
- Describe the presets by tradeoff:
  - Fast: lower runtime, more lenient cleanup
  - Balanced: default first pass
  - Quality: more markers, stricter cleanup, more retries

### 5. Scene setup wording overpromises floor detection

The setup operator supports selected-track floor alignment, but auto mode mostly delegates to Blender's standard scene setup.

Recommendation:

- Public docs should say:
  - "Create Blender tracking scene"
  - "Optionally select 3+ floor tracks before setup for floor alignment"
- Do not claim auto floor detection until it exists and is tested.

### 6. Test coverage is not yet product-grade

The package compiles, and there are a couple of benchmark scripts, but there is no real automated test suite for public behavior or Blender integration.

Recommendation:

- Add public-surface audit tests.
- Add ML dataset validation tests.
- Add at least one Blender/manual smoke checklist.
- Later add scripted Blender tests if CI can run Blender headless.

## Public Add-on Cleanup Plan

### Public behavior changes

Remove the following from the public Blender add-on:

- Research Beta panel
- `record_edits` scene property
- training-data export/import/share/reset/stats operators
- behavior monitoring after solve
- session recording during solve
- contributor ID generation during public add-on use
- automatic model updates from user sessions
- public references to telemetry, community data contribution, and AI learning

Keep or preserve:

- built-in presets
- footage type adjustments
- robust mode
- tripod mode
- region include/exclude tools
- failure diagnostics and heuristic retries
- smoothing
- scene setup
- old local data files on disk, but unused by the public add-on

### Public API removals

Remove these operator IDs:

- `autosolve.contribute_data`
- `autosolve.export_training_data`
- `autosolve.import_training_data`
- `autosolve.reset_training_data`
- `autosolve.view_training_stats`

Remove this scene property:

- `Scene.autosolve.record_edits`

### Runtime architecture after cleanup

The public add-on should become deterministic:

1. Read user settings from the UI.
2. Choose built-in settings from resolution, fps, footage type, quality preset, robust mode, and tripod mode.
3. Detect and distribute markers.
4. Track forward and backward.
5. Clean bad markers/tracks.
6. Heal or merge tracks only as an in-memory solve aid.
7. Solve with Blender.
8. Refine by filtering high-error tracks and retrying Blender solve.
9. Report solve status and error.

No data collection or model update should happen as part of this flow.

## Documentation Plan

### README.md

Rewrite around:

- Blender-native assisted camera tracking
- honest beta status
- feature list focused on current behavior
- clear workflow steps
- limitations and troubleshooting
- no data collection
- offline ML work as developer research only

Remove:

- public training-data contribution instructions
- HuggingFace upload links
- "AI improves over time" claims
- "rival industry giants" as a current claim
- future session-count promises for XGBoost/deep learning

### ARCHITECTURE.md

Update to describe:

- public add-on runtime pipeline
- deterministic preset/settings selection
- feature detection and cleanup phases
- failure diagnosis and retry logic
- no public data collection
- offline `ml/` tooling separation

Remove or rewrite:

- session recorder as public runtime component
- public learning loop diagrams
- export/import community model sections
- Research Beta UI references

### TRAINING_DATA.md and CONTRIBUTING_DATA.md

Replace with short legacy notes or remove from public docs.

Recommended replacement:

- "Public AutoSolve no longer collects training data."
- "Neural-network research now uses curated offline datasets in `ml/`."
- "See `ml/README.md` for developer-only dataset and training workflow."

## Offline Neural Network Plan

### Core strategy

Do not train a neural network to replace Blender's solver.

Train a neural network to choose better Blender-native tracking decisions.

This keeps AutoSolve inside the Blender ecosystem:

- Blender still detects, tracks, solves, and sets up the scene.
- The model only recommends settings or strategies.
- The public add-on does not ship PyTorch or external dependencies.
- Offline model outputs can later be distilled into reviewed defaults or small tables.

This is the realistic path to competing with stronger tools: make Blender start from better tracking choices and retry strategies.

### What the model should learn

The first useful model should be a reward predictor.

Instead of directly predicting one "best" setting, the model scores candidate settings for a clip. AutoSolve can later search over safe Blender-compatible candidates and choose the highest expected reward.

Inputs:

- clip metadata:
  - width
  - height
  - fps
  - frame count
  - footage type label
- motion and texture metrics:
  - feature density
  - average marker velocity
  - velocity variance
  - parallax proxy
  - dropout rate
  - blur/texture proxy if available
- candidate settings:
  - `pattern_size`
  - `search_size`
  - `correlation`
  - `threshold`
  - `motion_model`
  - robust/tripod flags

Outputs:

- expected solve reward
- optional success probability
- optional expected solve error
- optional expected bundle ratio

Reward should favor:

- valid solve
- low reprojection error
- high bundle ratio
- stable tracks
- reasonable runtime

Runtime should matter, but solve validity and quality should dominate.

### Curated dataset format

Use manually curated offline data, not public telemetry.

Each sample represents one controlled solve attempt:

```json
{
  "clip_id": "drone_beach_001",
  "clip": {
    "width": 3840,
    "height": 2160,
    "fps": 24,
    "frame_count": 180,
    "footage_type": "DRONE"
  },
  "features": {
    "feature_density": 0.42,
    "velocity_mean": 0.021,
    "velocity_std": 0.008,
    "parallax_score": 0.31,
    "dropout_rate": 0.18,
    "texture_score": 0.56
  },
  "settings": {
    "pattern_size": 55,
    "search_size": 231,
    "correlation": 0.62,
    "threshold": 0.22,
    "motion_model": "LocRot",
    "robust_mode": false,
    "tripod_mode": false
  },
  "result": {
    "success": true,
    "solve_error": 0.84,
    "bundle_count": 73,
    "track_count": 108,
    "runtime_seconds": 42.5
  }
}
```

The dataset split must be by `clip_id`, not random rows. This prevents the model from training and testing on different attempts from the same clip.

### `ml/` scripts

#### `prepare_dataset.py`

Command:

```bash
python ml/prepare_dataset.py --raw-dir ml/data/raw --out ml/data/processed/settings_dataset.pt
```

Responsibilities:

- read `.json`, `.jsonl`, and `.ndjson` curated trial files
- validate required fields
- validate setting ranges
- encode categorical fields
- compute reward
- split by `clip_id`
- normalize continuous features
- save a PyTorch dataset bundle

Outputs:

- processed tensors
- train/validation/test split indices
- normalization statistics
- category mappings
- validation report

#### `train_settings_model.py`

Command:

```bash
python ml/train_settings_model.py --data ml/data/processed/settings_dataset.pt --out ml/runs/settings_mlp
```

Responsibilities:

- load processed dataset
- train a small PyTorch MLP reward predictor
- use validation loss for early stopping
- save checkpoint and training metrics
- keep model small enough to be inspectable

Model target:

- primary: reward regression
- optional auxiliary heads:
  - success probability
  - solve error
  - bundle ratio

#### `evaluate_model.py`

Command:

```bash
python ml/evaluate_model.py --data ml/data/processed/settings_dataset.pt --checkpoint ml/runs/settings_mlp/best.pt --out ml/runs/eval.json
```

Responsibilities:

- evaluate prediction accuracy
- compare against built-in preset baselines
- report per-footage-type metrics
- report clips where model recommendations fail
- produce a concise JSON report

Important metrics:

- success-rate lift over built-in presets
- median solve error
- median bundle ratio
- reward improvement
- runtime impact
- failure breakdown by footage type

#### `export_defaults.py`

Command:

```bash
python ml/export_defaults.py --checkpoint ml/runs/settings_mlp/best.pt --out ml/runs/settings_defaults.json
```

Responsibilities:

- score a safe grid of Blender-compatible settings
- export best candidate settings by footage/motion class
- mark outputs as developer-reviewed defaults, not automatic runtime model integration

The public add-on should not auto-load this output in the current pass.

## How This Competes While Staying in Blender

Commercial tracking tools often win because they make better automatic decisions before and during the solve:

- better initial feature placement
- better search windows
- better handling of fast motion
- better rejection of bad tracks
- better retry strategies
- stronger defaults for footage types

AutoSolve can attack that same layer while keeping Blender as the engine.

The model does not need to solve camera geometry. Blender already does that.

The model needs to answer:

> For this clip and these candidate settings, how likely is Blender's native tracker/solver to produce a good solve?

Then AutoSolve can use Blender better than a generic default workflow.

This makes the project credible:

- no external solver dependency
- no PyTorch dependency in the add-on
- no user data collection in public builds
- measurable improvement through offline experiments
- future model outputs can be distilled into simple settings tables

## Test Plan

### Public add-on tests

- Run:

```bash
python -m compileall autosolve
```

- Add a public-surface audit test that checks:
  - no Research Beta panel
  - no public training-data operators
  - no public telemetry/contribution wording in README or add-on UI
  - no contributor ID usage from normal solve path

- Manual Blender smoke test:
  - enable add-on
  - load clip
  - run Auto-Track & Solve
  - confirm no session/model/behavior/contributor files are written
  - confirm removed operators do not appear in the UI

### ML tooling tests

- Run:

```bash
python -m compileall ml
```

- Unit tests for dataset validation:
  - valid NDJSON fixture
  - missing required clip field
  - invalid setting range
  - unsupported motion model
  - split by `clip_id`, not by row

- Training smoke test:
  - tiny fixture dataset
  - one epoch
  - checkpoint written
  - evaluation JSON written

## Acceptance Criteria

Public add-on cleanup is complete when:

- no telemetry/data-collection UI exists
- removed operator IDs are no longer registered
- `record_edits` is gone
- normal solves do not write training/session/behavior/contributor files
- docs no longer ask users to contribute tracking data
- docs no longer claim current AI/neural behavior
- quality preset docs match implementation
- scene setup docs do not claim auto floor detection

Offline ML foundation is complete when:

- `ml/` contains documented scripts
- curated trial data can be validated and converted to tensors
- a PyTorch MLP can train on the processed dataset
- evaluation compares model scoring against baseline presets
- exported defaults are clearly marked developer-review only

## Suggested Milestones

### Milestone 1: Public cleanup

- Remove public data collection from UI and operators.
- Remove automatic recording from the solve path.
- Rewrite public docs.
- Add audit test.

### Milestone 2: Offline dataset tooling

- Add `ml/prepare_dataset.py`.
- Add schema validation and fixture tests.
- Define reward function and split policy.

### Milestone 3: Offline model baseline

- Add reward-predictor MLP.
- Train on curated trial data.
- Evaluate against built-in presets.

### Milestone 4: Benchmark credibility

- Build a controlled benchmark clip set.
- Compare AutoSolve presets, ML-recommended settings, Blender defaults, and manual baseline where possible.
- Publish only measured claims.

## Final Recommendation

AutoSolve should not publicly present itself as an AI tracker today.

It should present itself as a Blender-native assisted tracking workflow today, while the neural-network optimizer is developed offline and validated with benchmarks.

That keeps user trust intact, keeps the add-on lightweight, and creates a credible path toward stronger automated solves without leaving Blender.
