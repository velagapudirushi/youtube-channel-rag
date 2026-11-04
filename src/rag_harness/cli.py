"""Command-Line Interface (CLI) for the standalone RAG Evaluation Harness (`rag-harness` / `rageval`)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from rag_harness.adapters import (
    CallablePipelineAdapter,
    HTTPPipelineAdapter,
    TraceFileAdapter,
    load_python_pipeline,
    load_question_set,
)
from rag_harness.demos.corpus import BENCHMARK_QUESTIONS, KNOWLEDGE_CORPUS
from rag_harness.demos.public_repos import PUBLIC_REPO_PIPELINES
from rag_harness.evaluator import RAGEvaluator
from rag_harness.metrics.ensemble import DualFaithfulnessEngine
from rag_harness.reporting.html_dashboard import generate_html_dashboard
from rag_harness.reporting.markdown import (
    generate_benchmark_comparison_markdown,
    generate_markdown_report,
)
from rag_harness.reporting.terminal import print_comparison_table, print_single_report
from rag_harness.schemas import HarnessReport, RetrievedChunk

app = typer.Typer(
    name="rag-harness",
    help=(
        "Standalone, framework-agnostic RAG evaluation CLI & library: "
        "Retrieval metrics (Recall@K, MRR, NDCG@K), Dual-Engine Faithfulness "
        "(LLM-Judge cross-checked against NLI Entailment), and Representative Failure Triage."
    ),
    add_completion=False,
)
console = Console()


@app.command("run")
def run_eval(
    pipeline: Optional[str] = typer.Option(
        None,
        "--pipeline",
        "-p",
        help="Built-in repo name OR Python module path (`my_app.py:run_rag`).",
    ),
    traces: Optional[Path] = typer.Option(
        None,
        "--traces",
        "-t",
        help="Path to pre-recorded RAG execution traces (`.jsonl` or `.json`).",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        "-e",
        help="HTTP webhook URL of a running RAG service (`POST /query`).",
    ),
    dataset: Optional[Path] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Path to question benchmark dataset (`.json` or `.jsonl`). Uses built-in 15-query set if omitted.",
    ),
    k: int = typer.Option(5, "--k", "-k", help="Top-K cutoff for retrieval metrics."),
    judge_model: str = typer.Option(
        "rubric-llm-judge-v1",
        "--judge-model",
        help="LLM Judge model identifier (e.g. `gpt-4o-mini` or `rubric-llm-judge-v1`).",
    ),
    nli_backend: str = typer.Option(
        "local",
        "--nli-backend",
        help="NLI Entailment backend (`local` structural cross-encoder or `transformers` DeBERTa-v3).",
    ),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Save JSON report to path."),
    output_md: Optional[Path] = typer.Option(None, "--output-md", help="Save Markdown report to path."),
    output_html: Optional[Path] = typer.Option(None, "--output-html", help="Save HTML dashboard to path."),
    save_traces: Optional[Path] = typer.Option(None, "--save-traces", help="Save raw execution traces as JSONL."),
    fail_under_faithfulness: float = typer.Option(
        0.0,
        "--fail-under-faithfulness",
        help="CI gate: exit with code 1 if cross-checked faithfulness is below this threshold (0.0-1.0).",
    ),
    fail_under_recall: float = typer.Option(
        0.0,
        "--fail-under-recall",
        help="CI gate: exit with code 1 if Recall@K is below this threshold (0.0-1.0).",
    ),
) -> None:
    """Evaluate a single RAG pipeline (via Python callable, JSONL trace file, or HTTP endpoint)."""
    questions = load_question_set(dataset) if dataset else BENCHMARK_QUESTIONS
    dataset_name = dataset.stem if dataset else "cloud_rag_benchmark_v1"

    if traces is not None:
        adapter = TraceFileAdapter(traces)
    elif endpoint is not None:
        adapter = HTTPPipelineAdapter(endpoint)
    elif pipeline is not None:
        if pipeline in PUBLIC_REPO_PIPELINES:
            inst = PUBLIC_REPO_PIPELINES[pipeline]()
            adapter = CallablePipelineAdapter(inst.query, pipeline_name=inst.pipeline_name)
        else:
            adapter = load_python_pipeline(pipeline)
    else:
        inst = PUBLIC_REPO_PIPELINES["langchain-naive-chroma-rag"]()
        adapter = CallablePipelineAdapter(inst.query, pipeline_name=inst.pipeline_name)

    evaluator = RAGEvaluator(
        k=k,
        judge_model=judge_model,
        nli_backend=nli_backend,
        cache_path=Path(".rag_harness_cache/judge_cache.sqlite"),
    )

    report = evaluator.run(
        adapter=adapter,
        questions=questions,
        dataset_name=dataset_name,
        trace_log_path=save_traces,
    )

    print_single_report(report, console=console)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]✓ Saved JSON report to {output_json}[/green]")

    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(generate_markdown_report(report), encoding="utf-8")
        console.print(f"[green]✓ Saved Markdown report to {output_md}[/green]")

    if output_html:
        generate_html_dashboard([report], output_html)
        console.print(f"[green]✓ Saved interactive HTML dashboard to {output_html}[/green]")

    if report.mean_cross_checked_faithfulness < fail_under_faithfulness:
        console.print(
            f"[bold red]CI Gate Failed:[/bold red] Cross-checked faithfulness "
            f"({report.mean_cross_checked_faithfulness:.1%}) < threshold ({fail_under_faithfulness:.1%})"
        )
        raise typer.Exit(code=1)

    if report.mean_recall_at_k < fail_under_recall:
        console.print(
            f"[bold red]CI Gate Failed:[/bold red] Recall@{k} "
            f"({report.mean_recall_at_k:.1%}) < threshold ({fail_under_recall:.1%})"
        )
        raise typer.Exit(code=1)


@app.command("benchmark")
def run_public_benchmarks(
    output_dir: Path = typer.Option(
        Path("benchmarks"),
        "--output-dir",
        "-o",
        help="Directory to write trace JSONLs, JSON reports, and Markdown summary.",
    ),
    docs_html: Path = typer.Option(
        Path("docs/index.html"),
        "--docs-html",
        help="Path for GitHub Pages interactive HTML dashboard.",
    ),
    k: int = typer.Option(5, "--k", "-k", help="Top-K retrieval cutoff."),
) -> None:
    """Run the harness against all 3 public RAG repo architectures and generate reports + GitHub Pages site."""
    evaluator = RAGEvaluator(
        k=k,
        cache_path=Path(".rag_harness_cache/judge_cache.sqlite"),
    )
    reports: List[HarnessReport] = []

    for repo_name, repo_cls in PUBLIC_REPO_PIPELINES.items():
        instance = repo_cls()
        adapter = CallablePipelineAdapter(instance.query, pipeline_name=repo_name)
        trace_path = output_dir / "traces" / f"{repo_name}.traces.jsonl"
        report = evaluator.run(
            adapter=adapter,
            questions=BENCHMARK_QUESTIONS,
            dataset_name="cloud_rag_benchmark_v1",
            trace_log_path=trace_path,
        )
        reports.append(report)

        report_json_path = output_dir / "reports" / f"{repo_name}.report.json"
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        report_md_path = output_dir / "reports" / f"{repo_name}.report.md"
        report_md_path.write_text(generate_markdown_report(report), encoding="utf-8")

    print_comparison_table(reports, console=console)

    # Also print detailed failure triage for the naive baseline so user sees immediate insights
    print_single_report(reports[0], console=console)

    comparison_md_path = output_dir / "BENCHMARK_REPORT.md"
    comparison_md_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_md_path.write_text(generate_benchmark_comparison_markdown(reports), encoding="utf-8")

    generate_html_dashboard(reports, docs_html)
    console.print(
        Panel(
            f"[bold green]✓ Benchmark Complete![/bold green]\n"
            f"• Trace Logs (JSONL):      [cyan]{output_dir / 'traces'}[/cyan]\n"
            f"• Markdown Report:         [cyan]{comparison_md_path}[/cyan]\n"
            f"• GitHub Pages Dashboard:  [cyan]{docs_html}[/cyan]",
            title="Artifacts Generated",
            border_style="green",
        )
    )


@app.command("check-claim")
def check_claim(
    context: str = typer.Option(..., "--context", "-c", help="Retrieved context text (premise)."),
    answer: str = typer.Option(..., "--answer", "-a", help="Generated RAG answer to decompose and verify."),
) -> None:
    """Decompose an answer into atomic claims and cross-check LLM-Judge vs. NLI Entailment inline."""
    engine = DualFaithfulnessEngine()
    chunk = RetrievedChunk(chunk_id="inline_chunk_1", text=context, score=1.0, rank=1)
    result = engine.evaluate(generated_answer=answer, retrieved_chunks=[chunk])

    console.print(
        Panel(
            f"[bold]LLM Judge Score:[/bold] {result.llm_judge_score:.2f}  |  "
            f"[bold]NLI Entailment Score:[/bold] {result.nli_entailment_score:.2f}  |  "
            f"[bold cyan]Cross-Checked Faithfulness:[/bold cyan] {result.cross_checked_score:.2f}\n"
            f"[bold yellow]Judge-NLI Disagreements:[/bold yellow] {result.disagreement_count}/{len(result.claims)}",
            title="Inline Dual-Judge Faithfulness Check",
            border_style="cyan",
        )
    )
    for idx, cl in enumerate(result.claims, start=1):
        status = (
            "[bold green]✓ AGREE[/bold green]"
            if cl.agreement
            else f"[bold red]⚡ DISAGREEMENT ({cl.disagreement_type})[/bold red]"
        )
        console.print(
            f"  [bold]{idx}. \"{cl.claim_text}\"[/bold] -> {status}\n"
            f"     LLM Judge: [cyan]{cl.llm_verdict.value}[/cyan] (conf={cl.llm_confidence:.2f}) | "
            f"NLI Model: [magenta]{cl.nli_label.value}[/magenta] "
            f"(E={cl.nli_probabilities.entailment:.2f}, N={cl.nli_probabilities.neutral:.2f}, "
            f"C={cl.nli_probabilities.contradiction:.2f}) | "
            f"Calibrated={cl.calibrated_score:.2f}\n"
            f"     [dim]{cl.llm_reasoning}[/dim]"
        )


@app.command("export-dataset")
def export_dataset(
    output: Path = typer.Option(
        Path("benchmarks/datasets/cloud_rag_benchmark.json"),
        "--output",
        "-o",
        help="Destination path for the exported benchmark question set.",
    ),
) -> None:
    """Export the built-in 15-query benchmark dataset and golden corpus to JSON."""
    import json

    output.parent.mkdir(parents=True, exist_ok=True)
    data = [q.model_dump(mode="json") for q in BENCHMARK_QUESTIONS]
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")

    corpus_path = output.parent / "knowledge_corpus.json"
    corpus_path.write_text(json.dumps(KNOWLEDGE_CORPUS, indent=2), encoding="utf-8")
    console.print(f"[green]✓ Exported {len(data)} benchmark questions to {output}[/green]")
    console.print(f"[green]✓ Exported {len(KNOWLEDGE_CORPUS)} corpus chunks to {corpus_path}[/green]")


if __name__ == "__main__":
    app()
