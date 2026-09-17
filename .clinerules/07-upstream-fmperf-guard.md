---
paths:
  - "fmperf/**"
  - "examples/**"
  - "docs/**"
  - "Dockerfile"
  - "Makefile"
  - "setup.py"
  - "scripts/**"
---

# 07 — Upstream fmperf guard

`fmperf/**` is a vendored copy of the upstream benchmark (fmperf-project/fmperf). `examples/**`,
`docs/SETUP.md`, `Dockerfile`, `Makefile` and `scripts/**` describe the upstream Kubernetes/kind/
OpenShift/energy flows. None of that infrastructure can be exercised in this environment.

## Rules

- Prefer not to modify `fmperf/**`; the MoST-specific behaviour belongs in `experiment_automation.py`
  and `requests/**`. When a change inside `fmperf/loadgen/run.py` is unavoidable (for example the
  runtime model discovery, the SaaS pacing path, or an env knob), keep it minimal, commented, and
  mention it in the commit message so the vendored diff stays auditable.
- `fmperf/utils/GpuCount.py` is a **shared interface**: its docstring states it is used by both this
  environment and the MoST API (`python GpuCount.py find --model ...`). Keep `GPU_COUNT_FIELD`, the
  subcommands (`find`, `from-csv`) and the JSON output shape stable; `requests/store_results.py`
  imports it as a best-effort dependency and must keep working when the import fails.
- `fmperf/loadgen/run.py` owns the load-generation contract (`results.json` per-token events with
  `worker_idx`/`request_idx`/`response_idx`/`n_tokens`/`timestamp`/`ok`/`error`/`duration_ms`).
  Every consumer in `requests/` assumes that schema: change it only with all consumers in mind.
- Leave `examples/**`, `docs/SETUP.md`, `Dockerfile`, `Makefile`, `requirements.txt` and
  `setup.py` alone unless the task is explicitly about the upstream stack. Note that
  `requirements-dev.txt` is referenced but missing, and that the `Makefile` format/lint/type-check
  targets need manually installed tools.
- Do not touch the legacy root-level `loadgen` file (stale copy of `fmperf/loadgen/run.py`), the
  binary fixtures (`fmperf/data/all_nbins_64.pkl`) or the energy test CSVs under
  `fmperf/loadgen/tests/`.
- Tests that exist here are `fmperf/tests/test_import.py` and `fmperf/loadgen/test_collect_energy.py`
  (run with `pytest fmperf/tests/` / `make test`); they do not cover the MoST harness, so covering new
  harness logic means adding small, self-contained checks as described in 05.
