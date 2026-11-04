"""Self-contained interactive HTML dashboard generator (GitHub Pages ready)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Union

from rag_harness.schemas import HarnessReport


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RAG Eval Harness — Interactive Benchmark & Failure Triage Report</title>
  <style>
    :root {
      --bg: #0b0f19;
      --panel: #131b2e;
      --panel-alt: #19233c;
      --border: #263354;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-2: #818cf8;
      --good: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
      --disagree: #f43f5e;
      --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      line-height: 1.55;
      padding: 28px 20px 60px;
    }
    .container { max-width: 1280px; margin: 0 auto; }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      padding-bottom: 22px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }
    .badge-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-family: var(--mono);
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.12);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
      margin-bottom: 8px;
    }
    h1 { font-size: 26px; font-weight: 700; letter-spacing: -0.02em; }
    .subtitle { color: var(--muted); font-size: 14px; max-width: 780px; margin-top: 4px; }
    .nav-tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 24px;
    }
    .tab-btn {
      background: var(--panel);
      color: var(--muted);
      border: 1px solid var(--border);
      padding: 9px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .tab-btn:hover { color: var(--text); border-color: var(--accent); }
    .tab-btn.active {
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border-color: var(--accent);
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .card-title {
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
    }
    th, td {
      padding: 11px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th {
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    tr:hover td { background: rgba(255, 255, 255, 0.02); }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }
    .kpi-box {
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
    }
    .kpi-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-val { font-size: 24px; font-weight: 700; font-family: var(--mono); margin: 4px 0; }
    .kpi-sub { font-size: 12px; color: var(--muted); }
    .good { color: var(--good); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .mono { font-family: var(--mono); }
    .bucket-card {
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-left: 4px solid var(--bad);
      border-radius: 10px;
      padding: 16px 18px;
      margin-bottom: 16px;
    }
    .bucket-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
    }
    .remediation-box {
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.3);
      color: #fde68a;
      padding: 9px 12px;
      border-radius: 6px;
      font-size: 13px;
      margin: 10px 0 14px;
    }
    .example-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
      margin-top: 10px;
      font-size: 13px;
    }
    .claim-row {
      margin-top: 8px;
      padding: 8px 10px;
      border-radius: 6px;
      background: rgba(255,255,255,0.03);
      border-left: 3px solid var(--border);
      font-size: 12.5px;
    }
    .claim-row.disagree {
      border-left-color: var(--disagree);
      background: rgba(244, 63, 94, 0.08);
    }
    .tag {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-family: var(--mono);
      font-weight: 600;
      margin-right: 6px;
    }
    .tag-good { background: rgba(34, 197, 94, 0.15); color: #4ade80; }
    .tag-bad { background: rgba(239, 68, 68, 0.18); color: #f87171; }
    .tag-warn { background: rgba(245, 158, 11, 0.18); color: #fbbf24; }
    .tag-disagree { background: rgba(244, 63, 94, 0.25); color: #fda4af; border: 1px solid #f43f5e; }
    .filter-bar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    input, textarea, select {
      width: 100%;
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      font-family: var(--font);
      font-size: 13.5px;
      margin-top: 6px;
      margin-bottom: 14px;
    }
    .btn-primary {
      background: var(--accent);
      color: #0b0f19;
      border: none;
      padding: 10px 18px;
      border-radius: 8px;
      font-weight: 700;
      cursor: pointer;
      font-size: 13.5px;
    }
    .btn-primary:hover { opacity: 0.92; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <div class="badge-pill">rag-eval-harness v0.2.0 • Dual-Engine Faithfulness (LLM-Judge × NLI Entailment)</div>
        <h1>Standalone RAG Evaluation Harness & Failure Triage</h1>
        <p class="subtitle">
          Framework-agnostic CLI & library evaluating Retrieval Metrics (Recall@K, MRR, NDCG@K), Cross-Checked Faithfulness
          (LLM-as-a-Judge verified against sentence-aligned NLI Entailment), and Representative Failure Taxonomy across public RAG architectures.
        </p>
      </div>
    </header>

    <div class="nav-tabs" id="mainTabs"></div>
    <div id="viewContainer"></div>
  </div>

  <script>
    const REPORTS = __REPORTS_JSON__;
    let activeTab = "overview";
    let traceFilter = "ALL";

    function scoreClass(v) {
      if (v >= 0.85) return "good";
      if (v >= 0.65) return "warn";
      return "bad";
    }

    function pct(v) { return (v * 100).toFixed(1) + "%"; }

    function renderTabs() {
      const nav = document.getElementById("mainTabs");
      let html = `<button class="tab-btn ${activeTab === 'overview' ? 'active' : ''}" onclick="switchTab('overview')">📊 Multi-Repo Comparison (${REPORTS.length})</button>`;
      REPORTS.forEach((r, idx) => {
        const key = "repo_" + idx;
        html += `<button class="tab-btn ${activeTab === key ? 'active' : ''}" onclick="switchTab('${key}')">🔍 ${r.pipeline_name} (${pct(r.pass_rate)})</button>`;
      });
      html += `<button class="tab-btn ${activeTab === 'sandbox' ? 'active' : ''}" onclick="switchTab('sandbox')">⚡ Live Dual-Judge Sandbox</button>`;
      nav.innerHTML = html;
    }

    function switchTab(tabKey) {
      activeTab = tabKey;
      traceFilter = "ALL";
      renderTabs();
      renderView();
    }

    function setTraceFilter(f) {
      traceFilter = f;
      renderView();
    }

    function renderOverview() {
      let rows = REPORTS.map((r, idx) => `
        <tr>
          <td><a href="javascript:void(0)" onclick="switchTab('repo_${idx}')" style="color:var(--accent);font-weight:600;text-decoration:none">${r.pipeline_name}</a></td>
          <td class="mono ${scoreClass(r.pass_rate)}">${pct(r.pass_rate)} (${r.passed_queries}/${r.total_queries})</td>
          <td class="mono ${scoreClass(r.mean_recall_at_k)}">${pct(r.mean_recall_at_k)}</td>
          <td class="mono ${scoreClass(r.mean_mrr)}">${r.mean_mrr.toFixed(3)}</td>
          <td class="mono ${scoreClass(r.mean_ndcg_at_k)}">${r.mean_ndcg_at_k.toFixed(3)}</td>
          <td class="mono">${pct(r.mean_llm_faithfulness)}</td>
          <td class="mono">${pct(r.mean_nli_faithfulness)}</td>
          <td class="mono ${scoreClass(r.mean_cross_checked_faithfulness)}" style="font-weight:700">${pct(r.mean_cross_checked_faithfulness)}</td>
          <td class="mono ${r.total_judge_disagreements > 0 ? 'warn' : 'good'}">${r.total_judge_disagreements}/${r.total_claims_evaluated} (${pct(r.disagreement_rate)})</td>
          <td class="mono">${r.mean_latency_ms.toFixed(1)} ms</td>
        </tr>
      `).join("");

      const cats = [
        "JUDGE_NLI_DISAGREEMENT",
        "INTRINSIC_CONTRADICTION",
        "EXTRINSIC_HALLUCINATION",
        "RETRIEVAL_MISS",
        "LOST_IN_THE_MIDDLE",
        "INCOMPLETE_SYNTHESIS"
      ];
      let failureRows = cats.map(c => `
        <tr>
          <td class="mono" style="color:var(--accent)">${c}</td>
          ${REPORTS.map(r => `<td class="mono">${r.failure_distribution[c] || 0}</td>`).join("")}
        </tr>
      `).join("");

      return `
        <div class="card">
          <div class="card-title">
            <span>Head-to-Head RAG Architecture Benchmark</span>
            <span class="mono" style="font-size:12px;color:var(--muted)">Click any pipeline to inspect traces & representative failures</span>
          </div>
          <div style="overflow-x:auto">
            <table>
              <thead>
                <tr>
                  <th>RAG Pipeline / Repo</th>
                  <th>Pass Rate</th>
                  <th>Recall@K</th>
                  <th>MRR</th>
                  <th>NDCG@K</th>
                  <th>LLM Judge Only</th>
                  <th>NLI Entailment</th>
                  <th>Cross-Checked Score</th>
                  <th>Judge-NLI Disagreements</th>
                  <th>Latency</th>
                </tr>
              </thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-title"><span>Why Cross-Check LLM-as-a-Judge Against NLI Entailment?</span></div>
          <p style="font-size:14px;color:var(--muted);margin-bottom:14px">
            Single-method LLM judges evaluate the whole retrieved context block holistically and suffer from <strong>keyword-halo bias</strong>:
            when a RAG answer swaps numbers or rules between two tiers mentioned in the same chunk (e.g., attributing a Standard plan's 30-day refund window to an Enterprise plan),
            passage-level LLM judges often approve it because both entities and numbers appear in the text. Cross-checking each atomic claim with a sentence-aligned
            <strong>NLI Entailment model</strong> catches polarity inversions, numeric mismatches, and scope swaps deterministically.
          </p>
          <div style="overflow-x:auto">
            <table>
              <thead>
                <tr>
                  <th>Root-Cause Failure Category</th>
                  ${REPORTS.map(r => `<th>${r.pipeline_name}</th>`).join("")}
                </tr>
              </thead>
              <tbody>${failureRows}</tbody>
            </table>
          </div>
        </div>
      `;
    }

    function renderRepoReport(r) {
      let bucketsHtml = r.representative_failures.length === 0
        ? `<p class="good">Zero failures detected across all ${r.total_queries} benchmark queries.</p>`
        : r.representative_failures.map(b => `
          <div class="bucket-card">
            <div class="bucket-header">
              <div>
                <span class="tag tag-bad">${b.category}</span>
                <strong style="font-size:15px">${b.title}</strong>
              </div>
              <span class="mono warn">${b.count} queries (${b.percentage.toFixed(1)}%)</span>
            </div>
            <p style="font-size:13.5px;color:var(--muted)">${b.description}</p>
            <div class="remediation-box"><strong>Recommended Engineering Fix:</strong> ${b.remediation}</div>
            ${b.examples.map((ex, i) => `
              <div class="example-box">
                <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:6px">
                  <strong>Representative Failure #${i+1} (${ex.query_id}): ${ex.question}</strong>
                  <span class="mono" style="font-size:12px">
                    LLM: ${ex.llm_judge_score.toFixed(2)} | NLI: ${ex.nli_entailment_score.toFixed(2)} |
                    <strong class="${scoreClass(ex.cross_checked_score)}">Cross-Checked: ${ex.cross_checked_score.toFixed(2)}</strong>
                  </span>
                </div>
                <div style="font-size:12.5px;color:var(--muted);margin-bottom:4px">
                  <strong>Retrieved Chunks:</strong> <span class="mono">${JSON.stringify(ex.retrieved_chunk_ids)}</span>
                  (Golden: <span class="mono">${JSON.stringify(ex.golden_chunk_ids)}</span>,
                  First Relevant Rank: <span class="mono">${ex.first_relevant_rank || 'MISSED'}</span>)
                </div>
                <div style="margin-bottom:4px"><strong>RAG Answer:</strong> ${ex.generated_answer}</div>
                <div style="color:var(--muted);margin-bottom:6px"><strong>Ground Truth:</strong> ${ex.ground_truth_answer}</div>
                <div style="color:#f87171;font-size:12.5px"><strong>Diagnosis:</strong> ${ex.failure_explanation}</div>
                ${ex.flagged_claims.map(fc => `
                  <div class="claim-row ${!fc.agreement ? 'disagree' : ''}">
                    <div><strong>Atomic Claim:</strong> "${fc.claim_text}"</div>
                    <div style="margin-top:4px">
                      <span class="tag ${fc.llm_verdict === 'SUPPORTED' ? 'tag-good' : 'tag-bad'}">LLM: ${fc.llm_verdict}</span>
                      <span class="tag ${fc.nli_label === 'ENTAILMENT' ? 'tag-good' : (fc.nli_label === 'CONTRADICTION' ? 'tag-bad' : 'tag-warn')}">
                        NLI: ${fc.nli_label} (E=${fc.nli_probabilities.entailment.toFixed(2)}, N=${fc.nli_probabilities.neutral.toFixed(2)}, C=${fc.nli_probabilities.contradiction.toFixed(2)})
                      </span>
                      ${!fc.agreement ? `<span class="tag tag-disagree">⚡ DISAGREEMENT: ${fc.disagreement_type}</span>` : ''}
                    </div>
                    <div style="font-size:12px;color:var(--muted);margin-top:4px">${fc.llm_reasoning}</div>
                  </div>
                `).join("")}
              </div>
            `).join("")}
          </div>
        `).join("");

      const filteredQueries = r.query_results.filter(q => {
        if (traceFilter === "ALL") return true;
        if (traceFilter === "FAILED") return !q.passed;
        if (traceFilter === "DISAGREE") return q.faithfulness.has_judge_disagreement;
        return q.failure_category === traceFilter;
      });

      let tracesHtml = filteredQueries.map(q => `
        <tr>
          <td class="mono">${q.query_id}</td>
          <td>
            <div style="font-weight:600">${q.question}</div>
            <div style="font-size:12.5px;color:var(--muted);margin-top:4px">${q.generated_answer}</div>
          </td>
          <td class="mono ${scoreClass(q.retrieval.recall_at_k)}">
            R@${r.k}=${q.retrieval.recall_at_k.toFixed(2)}<br/>
            MRR=${q.retrieval.mrr.toFixed(2)}
          </td>
          <td class="mono">
            LLM=${q.faithfulness.llm_judge_score.toFixed(2)}<br/>
            NLI=${q.faithfulness.nli_entailment_score.toFixed(2)}<br/>
            <strong class="${scoreClass(q.faithfulness.cross_checked_score)}">Cal=${q.faithfulness.cross_checked_score.toFixed(2)}</strong>
          </td>
          <td>
            <span class="tag ${q.passed ? 'tag-good' : 'tag-bad'}">${q.failure_category}</span>
            ${q.faithfulness.has_judge_disagreement ? '<span class="tag tag-disagree">LLM≠NLI</span>' : ''}
            <div style="font-size:12px;color:var(--muted);margin-top:4px">${q.failure_explanation}</div>
          </td>
        </tr>
      `).join("");

      return `
        <div class="metric-grid">
          <div class="kpi-box">
            <div class="kpi-label">Pass Rate</div>
            <div class="kpi-val ${scoreClass(r.pass_rate)}">${pct(r.pass_rate)}</div>
            <div class="kpi-sub">${r.passed_queries} of ${r.total_queries} queries passed</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">Recall@${r.k} / MRR</div>
            <div class="kpi-val ${scoreClass(r.mean_recall_at_k)}">${pct(r.mean_recall_at_k)}</div>
            <div class="kpi-sub">MRR: ${r.mean_mrr.toFixed(3)} • NDCG@${r.k}: ${r.mean_ndcg_at_k.toFixed(3)}</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">Cross-Checked Faithfulness</div>
            <div class="kpi-val ${scoreClass(r.mean_cross_checked_faithfulness)}">${pct(r.mean_cross_checked_faithfulness)}</div>
            <div class="kpi-sub">LLM: ${pct(r.mean_llm_faithfulness)} vs NLI: ${pct(r.mean_nli_faithfulness)}</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">Judge-NLI Disagreements</div>
            <div class="kpi-val ${r.total_judge_disagreements > 0 ? 'warn' : 'good'}">${r.total_judge_disagreements}/${r.total_claims_evaluated}</div>
            <div class="kpi-sub">${pct(r.disagreement_rate)} of atomic claims diverged</div>
          </div>
        </div>

        <div class="card">
          <div class="card-title"><span>Representative Failure Report (${r.pipeline_name})</span></div>
          ${bucketsHtml}
        </div>

        <div class="card">
          <div class="card-title">
            <span>Execution Trace Log (${filteredQueries.length} queries)</span>
          </div>
          <div class="filter-bar">
            <button class="tab-btn ${traceFilter==='ALL'?'active':''}" onclick="setTraceFilter('ALL')">All (${r.query_results.length})</button>
            <button class="tab-btn ${traceFilter==='FAILED'?'active':''}" onclick="setTraceFilter('FAILED')">Failed (${r.total_queries - r.passed_queries})</button>
            <button class="tab-btn ${traceFilter==='DISAGREE'?'active':''}" onclick="setTraceFilter('DISAGREE')">⚡ Judge-NLI Disagreements</button>
            <button class="tab-btn ${traceFilter==='RETRIEVAL_MISS'?'active':''}" onclick="setTraceFilter('RETRIEVAL_MISS')">Retrieval Miss</button>
            <button class="tab-btn ${traceFilter==='INTRINSIC_CONTRADICTION'?'active':''}" onclick="setTraceFilter('INTRINSIC_CONTRADICTION')">Contradictions</button>
          </div>
          <div style="overflow-x:auto">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Question & Generated Answer</th>
                  <th>Retrieval</th>
                  <th>Faithfulness</th>
                  <th>Root-Cause Verdict</th>
                </tr>
              </thead>
              <tbody>${tracesHtml}</tbody>
            </table>
          </div>
        </div>
      `;
    }

    function renderSandbox() {
      return `
        <div class="card">
          <div class="card-title"><span>⚡ Interactive Dual-Judge Faithfulness Sandbox (Client-Side Demo)</span></div>
          <p style="font-size:13.5px;color:var(--muted);margin-bottom:16px">
            Test how atomic claim decomposition + LLM-Judge vs. NLI Entailment cross-checking catches subtle entity swaps, numeric mismatches, and hallucinations.
          </p>
          <label style="font-size:12.5px;font-weight:600;color:var(--muted)">RETRIEVED CONTEXT CHUNK (PREMISE)</label>
          <textarea id="sbContext" rows="4">Standard tier subscriptions include a 30-day refund window and support up to 100 requests per minute. Enterprise tier contracts are strictly non-refundable and include dedicated 99.99% uptime SLA with 5,000 requests per minute.</textarea>

          <label style="font-size:12.5px;font-weight:600;color:var(--muted)">GENERATED RAG ANSWER (HYPOTHESIS TO DECOMPOSE & VERIFY)</label>
          <textarea id="sbAnswer" rows="3">Enterprise tier contracts include a 30-day refund window. Enterprise customers also receive a 99.99% uptime SLA and 5,000 requests per minute.</textarea>

          <button class="btn-primary" onclick="runBrowserSandbox()">Run Dual-Judge Cross-Check</button>
          <div id="sbOutput" style="margin-top:20px"></div>
        </div>
      `;
    }

    function runBrowserSandbox() {
      const ctx = document.getElementById("sbContext").value.trim();
      const ans = document.getElementById("sbAnswer").value.trim();
      const claims = ans.split(/(?<=[.!?])\\s+/).filter(Boolean);
      const ctxSentences = ctx.split(/(?<=[.!?])\\s+/).filter(Boolean);
      const ctxLower = ctx.toLowerCase();
      const ctxNums = (ctx.match(/\\$?\\d+(?:\\.\\d+)?%?/g) || []);

      let resultsHtml = claims.map((c, idx) => {
        const cLower = c.toLowerCase();
        const words = (cLower.match(/\\b[a-z0-9]{4,}\\b/g) || []);
        const matchedWords = words.filter(w => ctxLower.includes(w));
        const coverage = words.length ? matchedWords.length / words.length : 0;
        const cNums = (c.match(/\\$?\\d+(?:\\.\\d+)?%?/g) || []);
        const missingNums = cNums.filter(n => !ctxNums.includes(n));

        // Holistic LLM Judge check
        let llmVerdict = (coverage >= 0.55 && missingNums.length === 0) ? "SUPPORTED" : (missingNums.length > 0 ? "CONTRADICTED" : "UNVERIFIABLE");

        // Sentence-aligned NLI check
        let nliLabel = "NEUTRAL";
        let pEnt = 0.20, pNeut = 0.70, pContra = 0.10;
        let note = "No single premise sentence fully entails this claim.";

        const scopes = ["standard", "enterprise", "pro", "free"];
        const claimScopes = scopes.filter(s => cLower.includes(s));

        for (const s of ctxSentences) {
          const sLower = s.toLowerCase();
          const sScopes = scopes.filter(sc => sLower.includes(sc));
          const sNums = (s.match(/\\$?\\d+(?:\\.\\d+)?%?/g) || []);
          const sOverlap = words.filter(w => sLower.includes(w)).length / Math.max(words.length, 1);

          const antonymFlip = (cLower.includes("refund") && !cLower.includes("non-refund") && sLower.includes("non-refund"));
          const scopeMismatch = claimScopes.length && sScopes.length && !claimScopes.some(cs => sScopes.includes(cs));

          if (sOverlap >= 0.4 && !scopeMismatch && antonymFlip) {
            nliLabel = "CONTRADICTION";
            pContra = 0.94; pEnt = 0.02; pNeut = 0.04;
            note = `Polarity/scope contradiction against sentence: "${s}"`;
            break;
          }
          if (scopeMismatch && cNums.length && cNums.some(n => sNums.includes(n))) {
            nliLabel = "CONTRADICTION";
            pContra = 0.91; pEnt = 0.03; pNeut = 0.06;
            note = `Entity tier swap: number ${JSON.stringify(cNums)} belongs to [${sScopes.join(",")}] in "${s}", not [${claimScopes.join(",")}].`;
            break;
          }
          if (sOverlap >= 0.55 && !scopeMismatch && cNums.every(n => sNums.includes(n))) {
            nliLabel = "ENTAILMENT";
            pEnt = 0.93; pNeut = 0.05; pContra = 0.02;
            note = `Directly entailed by premise sentence: "${s}"`;
          }
        }

        const agree = (llmVerdict === "SUPPORTED" && nliLabel === "ENTAILMENT") ||
                      (llmVerdict === "CONTRADICTED" && nliLabel === "CONTRADICTION") ||
                      (llmVerdict === "UNVERIFIABLE" && nliLabel === "NEUTRAL");

        return `
          <div class="claim-row ${!agree ? 'disagree' : ''}">
            <div><strong>Claim #${idx+1}:</strong> "${c}"</div>
            <div style="margin-top:6px">
              <span class="tag ${llmVerdict==='SUPPORTED'?'tag-good':'tag-bad'}">LLM Judge: ${llmVerdict}</span>
              <span class="tag ${nliLabel==='ENTAILMENT'?'tag-good':(nliLabel==='CONTRADICTION'?'tag-bad':'tag-warn')}">
                NLI Model: ${nliLabel} (E=${pEnt.toFixed(2)}, N=${pNeut.toFixed(2)}, C=${pContra.toFixed(2)})
              </span>
              ${!agree ? '<span class="tag tag-disagree">⚡ JUDGE-NLI DISAGREEMENT (LLM Fooled by Tier Swap)</span>' : '<span class="tag tag-good">✓ Judges Agree</span>'}
            </div>
            <div style="font-size:12px;color:var(--muted);margin-top:4px">${note}</div>
          </div>
        `;
      }).join("");

      document.getElementById("sbOutput").innerHTML = resultsHtml;
    }

    function renderView() {
      const container = document.getElementById("viewContainer");
      if (activeTab === "overview") {
        container.innerHTML = renderOverview();
      } else if (activeTab === "sandbox") {
        container.innerHTML = renderSandbox();
        runBrowserSandbox();
      } else if (activeTab.startsWith("repo_")) {
        const idx = parseInt(activeTab.split("_")[1], 10);
        container.innerHTML = renderRepoReport(REPORTS[idx]);
      }
    }

    renderTabs();
    renderView();
  </script>
</body>
</html>
"""


def generate_html_dashboard(
    reports: Sequence[HarnessReport],
    output_path: Union[str, Path],
) -> Path:
    """Write a self-contained interactive HTML report for single or multi-repo runs."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    reports_data = [r.model_dump(mode="json") for r in reports]
    html = HTML_TEMPLATE.replace("__REPORTS_JSON__", json.dumps(reports_data))
    out.write_text(html, encoding="utf-8")
    return out
