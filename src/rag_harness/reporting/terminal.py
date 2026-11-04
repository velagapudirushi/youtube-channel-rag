"""Rich terminal reporter for single-pipeline evaluations and multi-repo benchmark comparisons."""

from __future__ import annotations

from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rag_harness.schemas import HarnessReport


def _score_color(val: float) -> str:
    if val >= 0.85:
        return "bold green"
    if val >= 0.65:
        return "bold yellow"
    return "bold red"


def print_single_report(report: HarnessReport, console: Console | None = None) -> None:
    """Render a comprehensive terminal report for a single RAG pipeline evaluation."""
    con = console or Console()

    header = Text()
    header.append(f"RAG EVAL HARNESS — {report.pipeline_name}\n", style="bold cyan")
    header.append(
        f"Dataset: {report.dataset_name} ({report.total_queries} queries, cutoff K={report.k})  |  "
        f"Pass Rate: {report.pass_rate:.1%} ({report.passed_queries}/{report.total_queries})  |  "
        f"Mean Latency: {report.mean_latency_ms:.1f} ms",
        style="dim",
    )
    con.print(Panel(header, border_style="cyan"))

    # 1. Retrieval & Faithfulness Summary Table
    metrics_table = Table(title="Retrieval & Dual-Engine Faithfulness Metrics", show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric Group", style="cyan")
    metrics_table.add_column("Metric", style="white")
    metrics_table.add_column("Score", justify="right")
    metrics_table.add_column("Diagnostic Interpretation", style="dim")

    metrics_table.add_row(
        "Retrieval",
        f"Recall@{report.k}",
        Text(f"{report.mean_recall_at_k:.1%}", style=_score_color(report.mean_recall_at_k)),
        "Fraction of golden evidence chunks found in top-K",
    )
    metrics_table.add_row(
        "Retrieval",
        f"MRR (Mean Reciprocal Rank)",
        Text(f"{report.mean_mrr:.3f}", style=_score_color(report.mean_mrr)),
        "1/rank of first golden chunk (1.0 = always rank #1)",
    )
    metrics_table.add_row(
        "Retrieval",
        f"NDCG@{report.k}",
        Text(f"{report.mean_ndcg_at_k:.3f}", style=_score_color(report.mean_ndcg_at_k)),
        "Ranking quality weighting higher ranks exponentially",
    )
    metrics_table.add_row(
        "Retrieval",
        "Context Fact Coverage",
        Text(f"{report.mean_context_coverage:.1%}", style=_score_color(report.mean_context_coverage)),
        "Reference key facts present in retrieved text",
    )
    metrics_table.add_section()
    metrics_table.add_row(
        "Faithfulness",
        "Method A: LLM-as-a-Judge",
        Text(f"{report.mean_llm_faithfulness:.1%}", style=_score_color(report.mean_llm_faithfulness)),
        "Passage-level semantic judge (can miss subtle scope/negation swaps)",
    )
    metrics_table.add_row(
        "Faithfulness",
        "Method B: NLI Entailment Model",
        Text(f"{report.mean_nli_faithfulness:.1%}", style=_score_color(report.mean_nli_faithfulness)),
        "Sentence-aligned NLI cross-encoder (strict premise -> hypothesis)",
    )
    metrics_table.add_row(
        "Faithfulness",
        "Cross-Checked Faithfulness",
        Text(f"{report.mean_cross_checked_faithfulness:.1%}", style=_score_color(report.mean_cross_checked_faithfulness)),
        "Calibrated ensemble penalizing NLI contradictions & unverified claims",
    )
    metrics_table.add_row(
        "Cross-Check",
        "Judge-NLI Disagreements",
        Text(
            f"{report.total_judge_disagreements}/{report.total_claims_evaluated} ({report.disagreement_rate:.1%})",
            style="bold yellow" if report.total_judge_disagreements > 0 else "bold green",
        ),
        "Claims where LLM Judge & NLI Entailment diverged",
    )
    con.print(metrics_table)

    # 2. Representative Failures Breakdown
    if not report.representative_failures:
        con.print(Panel("[bold green]Zero failures detected across all queries![/bold green]"))
        return

    con.print("\n[bold red]Representative Failure Triage Report[/bold red]")
    for bucket in report.representative_failures:
        bucket_title = (
            f"[{bucket.category.value}] {bucket.title} — "
            f"{bucket.count} queries ({bucket.percentage:.1f}%)"
        )
        lines: list[str] = [
            f"[italic]{bucket.description}[/italic]",
            f"[bold yellow]Remediation:[/bold yellow] {bucket.remediation}\n",
        ]
        for idx, ex in enumerate(bucket.examples, start=1):
            lines.append(
                f"  [bold cyan]Example {idx} ({ex.query_id}):[/bold cyan] {ex.question}\n"
                f"    • [dim]Retrieved Chunks:[/dim] {ex.retrieved_chunk_ids} (Golden: {ex.golden_chunk_ids}, "
                f"First Hit Rank: {ex.first_relevant_rank or 'MISSED'})\n"
                f"    • [dim]Generated Answer:[/dim] {ex.generated_answer}\n"
                f"    • [dim]Scores:[/dim] LLM-Judge={ex.llm_judge_score:.2f} | "
                f"NLI-Entailment={ex.nli_entailment_score:.2f} | "
                f"[bold]Cross-Checked={ex.cross_checked_score:.2f}[/bold]\n"
                f"    • [bold red]Root Cause:[/bold red] {ex.failure_explanation}"
            )
            for fc in ex.flagged_claims[:2]:
                dis_badge = f" [DISAGREEMENT: {fc.disagreement_type}]" if not fc.agreement else ""
                lines.append(
                    f"      ↳ Claim: \"{fc.claim_text}\" -> LLM={fc.llm_verdict.value}, "
                    f"NLI={fc.nli_label.value} (E={fc.nli_probabilities.entailment:.2f}, "
                    f"C={fc.nli_probabilities.contradiction:.2f}){dis_badge}"
                )
        con.print(Panel("\n".join(lines), title=bucket_title, border_style="red"))


def print_comparison_table(reports: Sequence[HarnessReport], console: Console | None = None) -> None:
    """Render a side-by-side comparison table across multiple RAG pipelines."""
    con = console or Console()
    table = Table(
        title="Multi-Repo RAG Evaluation Benchmark Comparison",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Pipeline / Repo", style="bold white")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Recall@K", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("NDCG@K", justify="right")
    table.add_column("LLM Judge", justify="right")
    table.add_column("NLI Model", justify="right")
    table.add_column("Cross-Checked", justify="right")
    table.add_column("Disagreements", justify="right")
    table.add_column("Latency", justify="right")

    for r in reports:
        table.add_row(
            r.pipeline_name,
            Text(f"{r.pass_rate:.1%}", style=_score_color(r.pass_rate)),
            Text(f"{r.mean_recall_at_k:.1%}", style=_score_color(r.mean_recall_at_k)),
            Text(f"{r.mean_mrr:.3f}", style=_score_color(r.mean_mrr)),
            Text(f"{r.mean_ndcg_at_k:.3f}", style=_score_color(r.mean_ndcg_at_k)),
            f"{r.mean_llm_faithfulness:.1%}",
            f"{r.mean_nli_faithfulness:.1%}",
            Text(
                f"{r.mean_cross_checked_faithfulness:.1%}",
                style=_score_color(r.mean_cross_checked_faithfulness),
            ),
            f"{r.total_judge_disagreements}/{r.total_claims_evaluated} ({r.disagreement_rate:.1%})",
            f"{r.mean_latency_ms:.1f} ms",
        )
    con.print(table)
