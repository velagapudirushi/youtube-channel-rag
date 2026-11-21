# RAG Evaluation Report: `youtube-channel-rag`

- **Dataset:** `youtube-channel-rag-benchmark-v1` (7 queries, cutoff `K=5`)
- **Overall Pass Rate:** **100.0%** (7/7)
- **Cross-Checked Faithfulness:** **99.0%** (LLM-Judge: `100.0%` vs. NLI-Entailment: `97.7%`)
- **Judge-NLI Disagreements Flagged:** `0` / `32` atomic claims (`0.0%`)
- **Mean Latency:** `2.5 ms`

## 1. Retrieval & Faithfulness Summary

| Category | Metric | Score | Notes |
| :--- | :--- | :---: | :--- |
| **Retrieval** | `Recall@5` | **100.0%** | Fraction of golden chunks retrieved in top-5 |
| **Retrieval** | `Precision@5` | `47.1%` | Signal-to-noise ratio in top-5 window |
| **Retrieval** | `MRR` | **1.000** | Reciprocal rank of first golden chunk |
| **Retrieval** | `NDCG@5` | **0.977** | Rank-aware retrieval quality |
| **Retrieval** | `Context Fact Coverage` | `100.0%` | Reference key facts present in retrieved text |
| **Faithfulness** | `Method A: LLM-as-a-Judge` | `100.0%` | Holistic passage-level judge |
| **Faithfulness** | `Method B: NLI Entailment` | `97.7%` | Sentence-aligned premise-hypothesis cross-encoder |
| **Faithfulness** | **`Cross-Checked Score`** | **99.0%** | Calibrated ensemble penalizing NLI contradictions |

## 2. Representative Failures

All queries passed retrieval and faithfulness checks.