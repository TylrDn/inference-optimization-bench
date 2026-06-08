---
name: build-dashboard
description: Invoke when building or modifying the Plotly Dash benchmark results dashboard. Use when the user asks to implement the reporting dashboard, visualize benchmark results, add charts, or update the web UI.
model: inherit
readonly: false
is_background: false
---

# Build Plotly Dash Benchmark Dashboard

## Objective

Create `reporting/dashboard.py` — an interactive Plotly Dash web application that loads benchmark result JSON/CSV files from `results/` and renders comparative performance charts (TTFT, throughput, latency percentiles) across all backends and models.

---

## Files to Create / Modify

### Create: `reporting/dashboard.py`

Full production implementation.

**Imports:**
```python
from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
APP_TITLE = "Inference Optimization Benchmark Dashboard"
```

**Data loading functions (all typed, all documented):**

```python
def load_results_df(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Load all benchmark JSON result files from results_dir into a single DataFrame.
    
    Expected JSON schema: run_id, timestamp, backend, model, config, metrics (dict).
    Returns empty DataFrame if no results found.
    
    Args:
        results_dir: Path to directory containing *_results.json files.
    Returns:
        DataFrame with columns: run_id, timestamp, backend, model,
          ttft_p50_ms, ttft_p95_ms, ttft_p99_ms,
          throughput_rps, latency_p50_ms, latency_p95_ms, latency_p99_ms,
          total_requests, failed_requests, concurrency.
    """
    ...

def get_available_models(df: pd.DataFrame) -> list[str]: ...
def get_available_backends(df: pd.DataFrame) -> list[str]: ...
def filter_df(df: pd.DataFrame, backends: list[str], models: list[str]) -> pd.DataFrame: ...
```

**Dashboard layout:**

Build a full `Dash` app with:

1. **Header**: "Inference Optimization Benchmark Dashboard" with subtitle showing results directory path and last-loaded timestamp.

2. **Sidebar controls:**
   - Multi-select dropdown: Backend filter (vllm, nim, triton, llamacpp) — default: all
   - Multi-select dropdown: Model filter — default: all
   - Refresh button (reloads from `results/` directory)
   - "Last updated" timestamp display

3. **Summary cards row** (4 KPI cards):
   - Best TTFT P50 (backend name)
   - Best Throughput (requests/sec)
   - Best Latency P99
   - Total benchmark runs loaded

4. **Tabs with charts:**

   **Tab 1: TTFT Comparison**
   - Grouped bar chart: X=backend, Y=TTFT P50/P95/P99 (ms), color=metric type
   - Line chart: TTFT P50 vs concurrency level, color=backend
   - Use `plotly.graph_objects.Figure` directly (not px) for grouped bars

   **Tab 2: Throughput**
   - Bar chart: throughput_rps by backend, grouped by model
   - Scatter: throughput vs concurrency, color=backend
   - Table showing top-5 runs by throughput

   **Tab 3: Latency Distribution**
   - Box plot: latency_p50/p95/p99 per backend (using go.Box)
   - Violin chart option via checklist

   **Tab 4: Quantization Comparison**
   - Grouped bar: TTFT by quantization type (awq/gptq/fp16/gguf), same model, same backend
   - Side-by-side quality vs speed (if accuracy CSV present in results/)

   **Tab 5: Raw Data**
   - `dash_table.DataTable` with all results, sortable columns, CSV export button

5. **Footer**: link to GitHub, last commit, environment info.

**App instantiation:**
```python
def create_app(results_dir: Path = RESULTS_DIR, debug: bool = False) -> Dash:
    """Create and configure the Dash application.
    
    Args:
        results_dir: Directory to load benchmark results from.
        debug: Enable Dash debug mode.
    Returns:
        Configured Dash app instance.
    """
    app = Dash(__name__, title=APP_TITLE, suppress_callback_exceptions=True)
    app.layout = build_layout(results_dir)
    register_callbacks(app, results_dir)
    return app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch benchmark dashboard")
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app = create_app(results_dir=args.results_dir, debug=args.debug)
    app.run(debug=args.debug, port=args.port)
```

**Callbacks to implement:**
- `update_charts(n_clicks, backends_filter, models_filter)` → updates all 5 tabs' figures
- `update_kpi_cards(n_clicks, backends_filter, models_filter)` → updates 4 KPI card values
- `export_csv(n_clicks)` → triggers CSV download of filtered DataFrame

**Styling:**
- Use Dash Bootstrap Components (`dbc`) for layout
- Color palette: NVIDIA green (`#76b900`) as accent, dark background (`#1a1a1a`)
- All charts use `plotly` dark template with NVIDIA green highlights
- Responsive layout (works at 1280px wide minimum)

---

### Modify: `reporting/__init__.py`

Add:
```python
from reporting.dashboard import create_app
```

---

### Create: `tests/test_dashboard.py`

```python
def test_load_results_df_empty_dir(tmp_path): ...
def test_load_results_df_single_result(tmp_path): ...
def test_load_results_df_multiple_backends(tmp_path): ...
def test_filter_df_by_backend(): ...
def test_get_available_models(): ...
def test_create_app_returns_dash_instance(): ...
def test_dashboard_layout_has_required_tabs(): ...
```

---

### Modify: `pyproject.toml`

Add to `[project.scripts]`:
```toml
bench-dashboard = "reporting.dashboard:main"
```

---

## Acceptance Criteria

- [ ] `python reporting/dashboard.py --results-dir results/ --port 8050` starts without errors
- [ ] Dashboard loads and displays "No results found" gracefully when `results/` is empty
- [ ] Dashboard loads correctly with 5+ JSON result files in `results/`
- [ ] All 5 tabs render without JavaScript errors
- [ ] `pytest tests/test_dashboard.py` passes (no server required for unit tests)
- [ ] `mypy --strict reporting/dashboard.py` exits 0
- [ ] `ruff check reporting/dashboard.py` exits 0
- [ ] KPI cards correctly compute best values across all backends
- [ ] CSV export downloads a valid CSV file
- [ ] Backend filter updates all charts simultaneously
