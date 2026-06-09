# inference-optimization-bench — Task Board

**Repo:** inference-optimization-bench
**Completion:** 100% COMPLETE
**Last Audit:** 2026-06-09
**Status:** Priority 1 production bar met — CI `--cov-fail-under=80`, root `docker-compose.yml`, workload + dashboard tests green.

---

## Cursor Subagents

| Subagent | Invoke | Task | Est. Time |
|---|---|---|---|
| `build-nim-backend.md` | `/build-nim-backend` | NIM inference backend + exponential backoff | 45 min |
| `build-workloads.md` | `/build-workloads` | Workload generators (single-turn, multi-turn, tool-calling) + structured JSON logging | 60 min |
| `build-dashboard.md` | `/build-dashboard` | Plotly Dash reporting dashboard (5 tabs, NVIDIA theme) | 45 min |
| `build-k8s-job.md` | `/build-k8s-job` | Kubernetes benchmark job YAML (GPU, PVC, init container) | 30 min |

---

## Priority 1 — CRITICAL

### [x] 1.1 NIM Inference Backend
**File:** `bench/backends/nim_backend.py`
**What:** Implement `NIMBackend` class extending `AbstractBackend`. Connect to NVIDIA NIM via `openai` Python SDK with `base_url=NIM_BASE_URL`. Measure TTFT via streaming (`stream=True`, record time-to-first-chunk with `time.perf_counter()`). Implement exponential backoff for 429 responses (3 retries: 2/4/8s). Raise `BackendUnavailableError` on 5xx and timeouts. Register in `BACKEND_MAP` in `bench/backends/__init__.py`.
**Acceptance Criteria:**
- `pytest tests/test_nim_backend.py` all green (mock all network)
- `mypy --strict bench/backends/nim_backend.py` exits 0
- `python bench/run_bench.py --backend nim --config configs/quick.yaml` writes JSON to `results/`
- TTFT uses `perf_counter()` streaming (not `time.time()`)

> **Subagent:** `/build-nim-backend`

---

### [x] 1.2 Workload Generators
**Files:** `bench/workloads/__init__.py`, `bench/workloads/base_workload.py`, `bench/workloads/single_turn.py`, `bench/workloads/multi_turn.py`, `bench/workloads/tool_calling.py`
**What:** Create `WorkloadBase` ABC with `WorkloadSample` dataclass. Implement three workloads:
- `SingleTurnWorkload`: 5 categories (CODE, SUMMARIZE, QA, MATH, CREATIVE), 20+ templates each, deterministic with seed.
- `MultiTurnWorkload`: 3 conversation categories (DEBUGGING, TUTORING, PLANNING), configurable `max_turns`.
- `ToolCallingWorkload`: 10 tool schemas (get_weather, search_web, execute_python, etc.), 30+ prompt→tool mappings.
**Acceptance Criteria:**
- `pytest tests/test_workloads.py` all green
- All workloads produce identical output for same seed
- `WORKLOAD_MAP["single_turn"]` importable from `bench.workloads`
- `mypy --strict bench/workloads/` exits 0

> **Subagent:** `/build-workloads`

---

### [x] 1.3 Plotly Dash Reporting Dashboard
**File:** `reporting/dashboard.py`
**What:** Build Plotly Dash web app loading benchmark JSON files from `results/`. Include: backend/model filter dropdowns, refresh button, 4 KPI cards (best TTFT/throughput/P99, run count), 5 tabs (TTFT Comparison, Throughput, Latency Distribution, Quantization Comparison, Raw Data). Dark theme with NVIDIA green accent (`#76b900`). Deployable via `python reporting/dashboard.py --port 8050`.
**Acceptance Criteria:**
- App starts without errors on empty `results/` dir
- All 5 tabs render with 5+ result files present
- CSV export downloads valid CSV
- `pytest tests/test_dashboard.py` all green
- `mypy --strict reporting/dashboard.py` exits 0

> **Subagent:** `/build-dashboard`

---

### [x] 1.4 Kubernetes Benchmark Job
**Files:** `deploy/k8s/bench-job.yaml`, `deploy/k8s/README.md`, `deploy/k8s/kustomization.yaml`
**What:** Multi-document YAML with: Namespace, ServiceAccount, ConfigMap (bench config inline), Secret template (NIM_API_KEY), PersistentVolumeClaim (10Gi), Job spec (GPU node selector, init container for NIM health check, `nvcr.io/nvidia/pytorch:24.03-py3` image, results PVC mount, `ttlSecondsAfterFinished: 86400`, `backoffLimit: 2`).
**Acceptance Criteria:**
- `kubectl apply --dry-run=client -f deploy/k8s/bench-job.yaml` exits 0
- GPU resource requests/limits are present (`nvidia.com/gpu: "1"`)
- Init container validates NIM API before main container starts
- Results PVC mounted at `/results`

> **Subagent:** `/build-k8s-job`

---

### [x] 1.5 Structured JSON Logging to results/
**Files:** `bench/run_bench.py`, `bench/metrics.py`
**What:** Every benchmark run writes a structured JSON log file to `results/{run_id}_{backend}_{timestamp}.json` containing: `run_id` (uuid4), `timestamp` (ISO8601), `backend`, `model`, `config`, full `metrics` dict (ttft_p50/p95/p99, throughput_rps, latency_p50/p95/p99, total_requests, failed_requests), and `raw_samples` array.
**Acceptance Criteria:**
- Running `run_bench.py` with any backend produces a valid JSON file in `results/`
- JSON validates against the schema in `.cursorrules`
- File is named `{run_id}_{backend}_{YYYYMMDD_HHMMSS}.json`
- No PII or secrets appear in the JSON output

> **Subagent:** `/build-workloads`

---

## Priority 2 — POLISH

### [ ] 2.1 Warmup Integration with New Backends
**File:** `bench/warmup.py`
**What:** Ensure `warmup.py` works with NIM and workload-based prompts. Add `--skip-warmup` CLI flag for development. Warmup should respect concurrency settings from config.
**Acceptance Criteria:**
- `warmup.py` imports `BACKEND_MAP` and `WORKLOAD_MAP` without circular imports
- `--skip-warmup` flag suppresses warmup and logs a WARNING
- Warmup errors are non-fatal (log WARNING, continue with benchmark)

### [ ] 2.2 Sweep Config for NIM
**File:** `configs/full_sweep.yaml`
**What:** Add complete NIM backend section with model list, concurrency levels, quantization note (NIM handles server-side). Ensure sweep.py can iterate over NIM × model × concurrency.
**Acceptance Criteria:**
- `python bench/sweep.py --config configs/full_sweep.yaml --backend nim --dry-run` prints sweep plan without running
- NIM section includes `meta/llama3-70b-instruct` and `meta/llama3-8b-instruct`

### [ ] 2.3 CI Integration Tests for NIM
**File:** `.github/workflows/ci.yml`
**What:** Add optional integration test job (runs only with `NIM_API_KEY` secret set). Runs `pytest tests/ -m integration --backend nim` against live NIM API.
**Acceptance Criteria:**
- Integration job is optional (does not block PRs if secret missing)
- Uses GitHub Actions `secrets.NIM_API_KEY` environment variable
- Uploads results artifacts to GitHub Actions

---

## Priority 3 — ENHANCEMENT

### [ ] 3.1 Results CSV Aggregation
**File:** `reporting/aggregate_results.py`
**What:** Script that reads all `results/*.json` files and aggregates into a single `results/summary.csv` with columns: run_id, date, backend, model, workload, concurrency, ttft_p50_ms, ttft_p95_ms, ttft_p99_ms, throughput_rps, latency_p99_ms. Used by the dashboard and report generator.
**Acceptance Criteria:**
- `python reporting/aggregate_results.py` writes `results/summary.csv`
- Handles missing/malformed JSON files gracefully (skips with WARNING log)
- CSV columns are stable across all backend types

### [ ] 3.2 Justfile Task Runner
**File:** `justfile` (repo root)
**What:** Add common tasks: `just bench` (quick config), `just sweep` (full sweep), `just report` (generate HTML report), `just dashboard` (start Dash app), `just test` (unit tests), `just test-integration` (integration tests), `just lint` (ruff + mypy), `just k8s-run` (apply k8s job).
**Acceptance Criteria:**
- All tasks documented in `just --list`
- `just test` runs in <60s
- `just lint` exits 0 on clean codebase
