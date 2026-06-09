"""
reporting/dashboard.py
~~~~~~~~~~~~~~~~~~~~~~~
Plotly Dash web application for visualising NVIDIA inference benchmark results.

The dashboard auto-loads all CSVs from the ``results/`` directory at startup
and renders four charts using an NVIDIA-branded dark theme:

1. P50 / P95 / P99 latency by backend (grouped bar chart)
2. Mean TTFT by backend (bar chart)
3. Throughput (tokens/sec) by backend (bar chart)
4. Total latency vs batch_size, one trace per backend (scatter chart)

Usage::

    python reporting/dashboard.py --results-dir ./results --port 8050

The app binds to ``0.0.0.0`` by default so it is reachable from outside a
container.  Set ``--debug`` to enable Dash hot-reload.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

# ---------------------------------------------------------------------------
# NVIDIA colour scheme
# ---------------------------------------------------------------------------

_BG_COLOR = "#1a1a1a"
_TEXT_COLOR = "#76b900"        # NVIDIA green
_ACCENT_COLOR = "#00a651"      # secondary green
_PANEL_BG = "#242424"
_GRID_COLOR = "#333333"
_FONT_FAMILY = "Helvetica Neue, Helvetica, Arial, sans-serif"

# Trace palette — enough distinct colours for up to 8 backends
_TRACE_COLORS = [
    "#76b900",  # NVIDIA green
    "#00a651",  # secondary green
    "#0098d4",  # blue
    "#f7941d",  # orange
    "#e40046",  # red
    "#8a2be2",  # violet
    "#00ced1",  # dark turquoise
    "#ffd700",  # gold
]

_LAYOUT_COMMON = dict(
    paper_bgcolor=_PANEL_BG,
    plot_bgcolor=_PANEL_BG,
    font=dict(color=_TEXT_COLOR, family=_FONT_FAMILY),
    xaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
    yaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
    legend=dict(bgcolor=_BG_COLOR, bordercolor=_GRID_COLOR, borderwidth=1),
    margin=dict(l=60, r=30, t=50, b=60),
)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> Optional[pd.DataFrame]:
    """Load and concatenate all benchmark CSV files from *results_dir*.

    Each CSV is expected to have at least the columns produced by
    ``bench/run_bench.py``:
    ``request_id, backend, batch_size, prompt_tokens, completion_tokens,
    ttft_ms, tpot_ms, total_latency_ms, tokens_per_sec, vram_used_mib``.

    Missing optional columns are filled with ``NaN``.

    Args:
        results_dir: Path to the directory containing ``.csv`` result files.

    Returns:
        A combined :class:`pandas.DataFrame`, or ``None`` if no CSVs were
        found.
    """
    pattern = os.path.join(results_dir, "*.csv")
    paths = glob.glob(pattern)
    if not paths:
        return None

    frames: list[pd.DataFrame] = []
    for path in sorted(paths):
        try:
            df = pd.read_csv(path)
            # Inject filename-derived metadata if columns are missing
            if "backend" not in df.columns:
                stem = os.path.splitext(os.path.basename(path))[0]
                df["backend"] = stem
            frames.append(df)
        except Exception as exc:
            print(f"[dashboard] Warning: could not read {path}: {exc}", file=sys.stderr)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def compute_latency_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute p50 / p95 / p99 of ``total_latency_ms`` grouped by backend.

    Args:
        df: Raw benchmark DataFrame containing ``total_latency_ms`` and
            ``backend`` columns.

    Returns:
        DataFrame with columns ``backend``, ``p50``, ``p95``, ``p99``.
    """
    return (
        df.groupby("backend")["total_latency_ms"]
        .quantile([0.50, 0.95, 0.99])
        .unstack(level=-1)
        .rename(columns={0.50: "p50", 0.95: "p95", 0.99: "p99"})
        .reset_index()
    )


def compute_mean_ttft(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean TTFT per backend.

    Args:
        df: Raw benchmark DataFrame.

    Returns:
        DataFrame with columns ``backend``, ``mean_ttft_ms``.
    """
    return (
        df.groupby("backend")["ttft_ms"]
        .mean()
        .reset_index()
        .rename(columns={"ttft_ms": "mean_ttft_ms"})
    )


def compute_mean_throughput(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean tokens/sec per backend.

    Args:
        df: Raw benchmark DataFrame.

    Returns:
        DataFrame with columns ``backend``, ``mean_tps``.
    """
    return (
        df.groupby("backend")["tokens_per_sec"]
        .mean()
        .reset_index()
        .rename(columns={"tokens_per_sec": "mean_tps"})
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_latency_percentile_chart(df: pd.DataFrame) -> go.Figure:
    """Build a grouped bar chart of P50/P95/P99 latency by backend.

    Args:
        df: Raw benchmark DataFrame.

    Returns:
        Plotly :class:`~plotly.graph_objects.Figure`.
    """
    pct = compute_latency_percentiles(df)
    backends = pct["backend"].tolist()
    color_map = {b: _TRACE_COLORS[i % len(_TRACE_COLORS)] for i, b in enumerate(backends)}

    fig = go.Figure()
    for percentile, label in [("p50", "P50"), ("p95", "P95"), ("p99", "P99")]:
        fig.add_trace(
            go.Bar(
                name=label,
                x=pct["backend"],
                y=pct[percentile],
                marker_color=[color_map[b] for b in pct["backend"]],
                opacity=0.7 if label != "P50" else 1.0,
            )
        )

    fig.update_layout(
        **_LAYOUT_COMMON,
        title=dict(text="Latency Percentiles by Backend (ms)", font=dict(color=_TEXT_COLOR)),
        barmode="group",
        xaxis_title="Backend",
        yaxis_title="Latency (ms)",
    )
    return fig


def build_ttft_chart(df: pd.DataFrame) -> go.Figure:
    """Build a bar chart of mean TTFT by backend.

    Args:
        df: Raw benchmark DataFrame.

    Returns:
        Plotly :class:`~plotly.graph_objects.Figure`.
    """
    ttft = compute_mean_ttft(df)
    backends = ttft["backend"].tolist()
    colors = [_TRACE_COLORS[i % len(_TRACE_COLORS)] for i in range(len(backends))]

    fig = go.Figure(
        go.Bar(
            x=ttft["backend"],
            y=ttft["mean_ttft_ms"],
            marker_color=colors,
        )
    )
    fig.update_layout(
        **_LAYOUT_COMMON,
        title=dict(text="Mean Time-to-First-Token (TTFT) by Backend (ms)", font=dict(color=_TEXT_COLOR)),  # noqa: E501
        xaxis_title="Backend",
        yaxis_title="Mean TTFT (ms)",
    )
    return fig


def build_throughput_chart(df: pd.DataFrame) -> go.Figure:
    """Build a bar chart of mean throughput (tokens/sec) by backend.

    Args:
        df: Raw benchmark DataFrame.

    Returns:
        Plotly :class:`~plotly.graph_objects.Figure`.
    """
    tps = compute_mean_throughput(df)
    backends = tps["backend"].tolist()
    colors = [_TRACE_COLORS[i % len(_TRACE_COLORS)] for i in range(len(backends))]

    fig = go.Figure(
        go.Bar(
            x=tps["backend"],
            y=tps["mean_tps"],
            marker_color=colors,
        )
    )
    fig.update_layout(
        **_LAYOUT_COMMON,
        title=dict(text="Mean Throughput by Backend (tokens/sec)", font=dict(color=_TEXT_COLOR)),
        xaxis_title="Backend",
        yaxis_title="Tokens / second",
    )
    return fig


def build_latency_vs_batch_chart(df: pd.DataFrame) -> go.Figure:
    """Build a scatter chart of total latency vs batch_size, one trace per backend.

    If the DataFrame does not contain a ``batch_size`` column the chart is
    built against ``request_id`` as a fallback x-axis.

    Args:
        df: Raw benchmark DataFrame.

    Returns:
        Plotly :class:`~plotly.graph_objects.Figure`.
    """
    x_col = "batch_size" if "batch_size" in df.columns else "request_id"
    backends = sorted(df["backend"].unique())
    fig = go.Figure()

    for i, backend in enumerate(backends):
        sub = df[df["backend"] == backend]
        color = _TRACE_COLORS[i % len(_TRACE_COLORS)]

        # Aggregate mean latency per batch_size for cleaner visualisation
        if x_col == "batch_size":
            agg = sub.groupby(x_col)["total_latency_ms"].mean().reset_index()
            x_vals = agg[x_col]
            y_vals = agg["total_latency_ms"]
        else:
            x_vals = sub[x_col]
            y_vals = sub["total_latency_ms"]

        fig.add_trace(
            go.Scatter(
                name=backend,
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(color=color, size=7),
            )
        )

    x_title = "Batch Size" if x_col == "batch_size" else "Request ID"
    fig.update_layout(
        **_LAYOUT_COMMON,
        title=dict(text="Total Latency vs Batch Size (ms)", font=dict(color=_TEXT_COLOR)),
        xaxis_title=x_title,
        yaxis_title="Mean Total Latency (ms)",
    )
    return fig


# ---------------------------------------------------------------------------
# Empty / no-data figure
# ---------------------------------------------------------------------------

def _empty_figure(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_LAYOUT_COMMON,
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color=_TEXT_COLOR, size=16),
            )
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Dash app factory
# ---------------------------------------------------------------------------

def create_app(results_dir: str) -> dash.Dash:
    """Construct and configure the Dash application.

    Args:
        results_dir: Path to the directory from which CSVs are loaded.

    Returns:
        A configured :class:`dash.Dash` application instance (not yet running).
    """
    df = load_results(results_dir)
    no_data = df is None

    app = dash.Dash(
        __name__,
        title="NVIDIA Inference Benchmark Dashboard",
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    # ---- Layout ----
    header = html.Div(
        children=[
            html.H1(
                "NVIDIA Inference Benchmark Dashboard",
                style={
                    "color": _TEXT_COLOR,
                    "fontFamily": _FONT_FAMILY,
                    "fontSize": "28px",
                    "fontWeight": "bold",
                    "marginBottom": "4px",
                },
            ),
            html.P(
                f"Results directory: {os.path.abspath(results_dir)}",
                style={"color": "#aaaaaa", "fontFamily": _FONT_FAMILY, "fontSize": "13px"},
            ),
        ],
        style={"padding": "24px 32px 8px 32px", "backgroundColor": _BG_COLOR},
    )

    if no_data:
        body = html.Div(
            children=[
                html.P(
                    "No benchmark results found. Run bench/run_bench.py first.",
                    style={
                        "color": _TEXT_COLOR,
                        "fontFamily": _FONT_FAMILY,
                        "fontSize": "18px",
                        "textAlign": "center",
                        "marginTop": "120px",
                    },
                )
            ],
            style={"backgroundColor": _BG_COLOR, "minHeight": "80vh"},
        )
    else:
        chart_style = {
            "backgroundColor": _PANEL_BG,
            "borderRadius": "8px",
            "padding": "16px",
            "marginBottom": "24px",
        }
        body = html.Div(
            children=[
                # Row 1: latency percentiles + TTFT
                html.Div(
                    children=[
                        html.Div(
                            dcc.Graph(
                                id="latency-pct-chart",
                                figure=build_latency_percentile_chart(df),
                                config={"displayModeBar": False},
                            ),
                            style={**chart_style, "flex": "1", "marginRight": "12px"},
                        ),
                        html.Div(
                            dcc.Graph(
                                id="ttft-chart",
                                figure=build_ttft_chart(df),
                                config={"displayModeBar": False},
                            ),
                            style={**chart_style, "flex": "1", "marginLeft": "12px"},
                        ),
                    ],
                    style={"display": "flex", "flexDirection": "row"},
                ),
                # Row 2: throughput + latency vs batch size
                html.Div(
                    children=[
                        html.Div(
                            dcc.Graph(
                                id="throughput-chart",
                                figure=build_throughput_chart(df),
                                config={"displayModeBar": False},
                            ),
                            style={**chart_style, "flex": "1", "marginRight": "12px"},
                        ),
                        html.Div(
                            dcc.Graph(
                                id="latency-batch-chart",
                                figure=build_latency_vs_batch_chart(df),
                                config={"displayModeBar": False},
                            ),
                            style={**chart_style, "flex": "1", "marginLeft": "12px"},
                        ),
                    ],
                    style={"display": "flex", "flexDirection": "row"},
                ),
            ],
            style={"backgroundColor": _BG_COLOR, "padding": "16px 32px"},
        )

    app.layout = html.Div(
        children=[header, body],
        style={"backgroundColor": _BG_COLOR, "minHeight": "100vh"},
    )

    return app


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NVIDIA Inference Benchmark Dashboard — Plotly Dash web app."
    )
    parser.add_argument(
        "--results-dir",
        default="./results",
        help="Directory containing benchmark CSV files (default: ./results).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="TCP port to bind to (default: 8050).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug / hot-reload mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for the dashboard server.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    args = _parse_args(argv)
    app = create_app(results_dir=args.results_dir)
    print(
        f"[dashboard] Starting server at http://{args.host}:{args.port} "
        f"(results_dir={args.results_dir!r}, debug={args.debug})"
    )
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
