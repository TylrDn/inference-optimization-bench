"""Additional dashboard coverage for CLI and empty-state helpers."""

from __future__ import annotations

import argparse
from argparse import Namespace
from unittest.mock import MagicMock, patch

from reporting import dashboard


def test_empty_figure_renders_message() -> None:
    fig = dashboard._empty_figure("No benchmark data")
    assert fig.layout.annotations[0].text == "No benchmark data"


def test_parse_args_defaults() -> None:
    args = dashboard._parse_args([])
    assert args.results_dir == "./results"
    assert args.port == 8050


def test_main_starts_app() -> None:
    mock_app = MagicMock()
    args = Namespace(
        results_dir="./results",
        port=8050,
        host="127.0.0.1",
        debug=False,
    )
    with patch.object(dashboard, "create_app", return_value=mock_app):
        with patch.object(dashboard, "_parse_args", return_value=args):
            dashboard.main([])
    mock_app.run.assert_called_once_with(host="127.0.0.1", port=8050, debug=False)
