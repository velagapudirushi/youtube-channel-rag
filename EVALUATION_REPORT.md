# RAG Evaluation Report: `youtube-channel-rag`

- **Dataset:** `youtube-channel-rag-benchmark-v1` (7 queries, cutoff `K=5`)
- **Overall Pass Rate:** **100.0%** (7/7)
- **Cross-Checked Faithfulness:** **98.0%** (LLM-Judge: `97.1%` vs. NLI-Entailment: `98.0%`)
- **Judge-NLI Disagreements Flagged:** `1` / `31` atomic claims (`3.2%`)
- **Mean Latency:** `0.7 ms`

## 1. Retrieval & Faithfulness Summary

| Category | Metric | Score | Notes |
| :--- | :--- | :---: | :--- |
| **Retrieval** | `Recall@5` | **100.0%** | Fraction of golden chunks retrieved in top-5 |
| **Retrieval** | `Precision@5` | `100.0%` | Signal-to-noise ratio in top-5 window |
| **Retrieval** | `MRR` | **1.000** | Reciprocal rank of first golden chunk |
| **Retrieval** | `NDCG@5` | **1.000** | Rank-aware retrieval quality |
| **Retrieval** | `Context Fact Coverage` | `100.0%` | Reference key facts present in retrieved text |
| **Faithfulness** | `Method A: LLM-as-a-Judge` | `97.1%` | Holistic passage-level judge |
| **Faithfulness** | `Method B: NLI Entailment` | `98.0%` | Sentence-aligned premise-hypothesis cross-encoder |
| **Faithfulness** | **`Cross-Checked Score`** | **98.0%** | Calibrated ensemble penalizing NLI contradictions |

## 2. Representative Failures

All queries passed retrieval and faithfulness checks.