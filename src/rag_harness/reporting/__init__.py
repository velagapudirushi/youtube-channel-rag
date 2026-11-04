"""Reporting utilities for Terminal (Rich), GitHub Markdown, and Interactive HTML Dashboards."""

from rag_harness.reporting.html_dashboard import generate_html_dashboard
from rag_harness.reporting.markdown import (
    generate_benchmark_comparison_markdown,
    generate_markdown_report,
)
from rag_harness.reporting.terminal import print_comparison_table, print_single_report

__all__ = [
    "generate_html_dashboard",
    "generate_benchmark_comparison_markdown",
    "generate_markdown_report",
    "print_comparison_table",
    "print_single_report",
]
