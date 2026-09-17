# 06 — Git, artifacts and documentation upkeep

## Never commit

- `results/` and `results*` (experiment outputs), per-iteration `output.csv`, `first_half.csv`,
  `second_half.csv`, `prompts.csv`, `results.json`, `results_all.json`.
- `.env` (local configuration), `*.out` Slurm logs, `logs/`, `pod_logs*/`, `test_logs.log`.
- Generated workloads and prompt dumps (`*.jsonl`, `sample_requests*.json` outside the repo).
- `examples/*.csv`, `examples/*.json` (example outputs). `.gitignore` already covers these; keep
  `.gitignore` and `.clineignore` in sync when a new artifact type appears.

## Commit style

Match the existing history: a short summary line, past tense or imperative, one logical change per
commit — e.g. "Implemented GPU count and included it in results", "Bug fix: MIT not updating largest
true and smallest false correctly", "Fix: generated request file name did not coincide with sought
request file name", "Update .env.example", "Removed legacy files".

If a change touches both sides of a contract (see 02: argv positions, `results.csv` columns, scraped
stdout strings, folder naming), land both sides in the same commit — otherwise the pipeline is
silently broken between commits.

## The sync set

When a configuration knob, output column or folder name changes, update all of:

1. `.env.example` (documentation + default),
2. `load_env_config()` defaults in `experiment_automation.py`,
3. the `_get_*()` fallback literals,
4. `README.md` (setup notes and "Relevant `results.csv` fields"),
5. these rules (02/03) when the contract itself changes.

When a `results.csv` column changes: `store_results.main()` header → data row → README → any MoST
API/dashboard consumer (`fmperf/utils/GpuCount.py` reads `GPU_COUNT` by name and is shared with the
MoST API).

## Stale documentation to fix rather than propagate

- `README.md`: the archive folder is `Experiment_<EXPERIMENT_TYPE>_<timestamp>` (the README omits the
  `Experiment_` prefix), and results live under `RESULTS_DIR` (the README says `/requests`).
- `.env.example`: `THRESHOLD_TYPE`, `SUCCESS_RATE` and `PROMPTS_FILE` are documented but not read by
  this pipeline (`SUCCESS_RATE_THRESHOLD` and the hardcoded prompts filename are what the code uses).
- `MIT_PLATEAU_REL_TOL` / `MIT_PLATEAU_ABS_TOL` are implemented but undocumented in `.env.example`.
- The code default of `STOP_THRESHOLD` (0.5) differs from the shipped value (0.05).

## Repository hygiene

- Keep `experiment_setting.txt` (it documents a full 64-combination grid with per-combination
  `REQ_MIN_START`) unless the user asks to remove it.
- Do not delete or "modernize" the vendored `fmperf/**` tree as part of unrelated work (see 07).
- Prefer `git mv`/`git rm` for tracked moves so history stays readable, and check
  `git status --short` before concluding that new artifacts are expected.
