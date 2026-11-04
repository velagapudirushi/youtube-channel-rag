"""Dual-Engine Faithfulness Cross-Checker: LLM-as-a-Judge cross-checked against NLI Entailment."""

from __future__ import annotations

from typing import List, Optional, Sequence

from rag_harness.metrics.claims import decompose_into_claims, is_abstention_response
from rag_harness.metrics.llm_judge import LLMJudgeEvaluator
from rag_harness.metrics.nli_judge import NLIEvaluator
from rag_harness.schemas import (
    ClaimEvaluation,
    ClaimVerdict,
    FaithfulnessResult,
    NLILabel,
    RetrievedChunk,
)


class DualFaithfulnessEngine:
    """Cross-checks an LLM Judge against an NLI Entailment model at the atomic claim level."""

    def __init__(
        self,
        llm_judge: Optional[LLMJudgeEvaluator] = None,
        nli_judge: Optional[NLIEvaluator] = None,
    ) -> None:
        self.llm_judge = llm_judge or LLMJudgeEvaluator()
        self.nli_judge = nli_judge or NLIEvaluator()

    def evaluate(
        self,
        generated_answer: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> FaithfulnessResult:
        """Decompose `generated_answer` into atomic claims and cross-check LLM Judge vs NLI."""
        if is_abstention_response(generated_answer):
            return FaithfulnessResult(
                llm_judge_score=1.0,
                nli_entailment_score=1.0,
                cross_checked_score=1.0,
                claims=[],
                disagreement_count=0,
                has_judge_disagreement=False,
                contradiction_count=0,
                unverifiable_count=0,
                is_abstention=True,
            )

        raw_claims = decompose_into_claims(generated_answer)
        if not raw_claims:
            return FaithfulnessResult(
                llm_judge_score=0.0,
                nli_entailment_score=0.0,
                cross_checked_score=0.0,
                claims=[],
            )

        evaluated_claims: List[ClaimEvaluation] = []
        llm_scores: List[float] = []
        nli_scores: List[float] = []
        calibrated_scores: List[float] = []
        disagreement_count = 0
        contradiction_count = 0
        unverifiable_count = 0

        for claim_text in raw_claims:
            llm_out = self.llm_judge.evaluate_claim(claim_text, retrieved_chunks)
            nli_out = self.nli_judge.score_claim(claim_text, retrieved_chunks)

            # Map LLM verdict to raw score
            if llm_out.verdict == ClaimVerdict.SUPPORTED:
                llm_raw = 1.0
            elif llm_out.verdict == ClaimVerdict.UNVERIFIABLE:
                llm_raw = 0.15
            else:
                llm_raw = 0.0

            # Map NLI probabilities to continuous entailment score (penalizing contradiction)
            p_ent = nli_out.probabilities.entailment
            p_contra = nli_out.probabilities.contradiction
            if nli_out.label == NLILabel.ENTAILMENT:
                nli_raw = round(min(1.0, max(0.75, p_ent)), 4)
            elif nli_out.label == NLILabel.CONTRADICTION:
                nli_raw = 0.0
            else:
                nli_raw = round(max(0.0, p_ent * 0.5), 4)

            # Cross-check agreement and calibrate
            agreement = False
            disagreement_type: Optional[str] = None

            if llm_out.verdict == ClaimVerdict.SUPPORTED and nli_out.label == NLILabel.ENTAILMENT:
                agreement = True
                calibrated = round(0.55 * llm_raw + 0.45 * nli_raw, 4)
            elif llm_out.verdict == ClaimVerdict.CONTRADICTED and nli_out.label == NLILabel.CONTRADICTION:
                agreement = True
                calibrated = 0.0
                contradiction_count += 1
            elif llm_out.verdict == ClaimVerdict.UNVERIFIABLE and nli_out.label == NLILabel.NEUTRAL:
                agreement = True
                calibrated = round(0.20 * p_ent, 4)
                unverifiable_count += 1
            elif llm_out.verdict == ClaimVerdict.SUPPORTED and nli_out.label == NLILabel.CONTRADICTION:
                # Critical catch: LLM Judge rubber-stamped a subtle contradiction / scope swap!
                agreement = False
                disagreement_type = "LLM_MISSED_CONTRADICTION"
                disagreement_count += 1
                contradiction_count += 1
                calibrated = round(max(0.0, 0.12 * (1.0 - p_contra)), 4)
            elif llm_out.verdict == ClaimVerdict.SUPPORTED and nli_out.label == NLILabel.NEUTRAL:
                agreement = False
                disagreement_count += 1
                if p_contra <= 0.08 and llm_out.confidence >= 0.80:
                    disagreement_type = "CONCEPTUAL_PARAPHRASE_DIVERGENCE"
                    calibrated = round(0.55 + 0.35 * p_ent, 4)
                else:
                    disagreement_type = "LLM_OVERCONFIDENT_UNVERIFIED"
                    unverifiable_count += 1
                    calibrated = round(0.25 + 0.35 * p_ent, 4)
            elif llm_out.verdict in (ClaimVerdict.CONTRADICTED, ClaimVerdict.UNVERIFIABLE) and nli_out.label == NLILabel.ENTAILMENT:
                agreement = False
                disagreement_type = "LLM_FALSE_NEGATIVE"
                disagreement_count += 1
                calibrated = round(0.60 * p_ent, 4)
            else:
                # One says CONTRADICTED, the other says NEUTRAL
                agreement = False
                disagreement_type = "SEVERITY_DIVERGENCE"
                disagreement_count += 1
                if llm_out.verdict == ClaimVerdict.CONTRADICTED or nli_out.label == NLILabel.CONTRADICTION:
                    contradiction_count += 1
                else:
                    unverifiable_count += 1
                calibrated = 0.05

            reasoning_combined = llm_out.reasoning
            if not agreement and nli_out.diagnostic_note:
                reasoning_combined = f"{llm_out.reasoning} [NLI Cross-Check: {nli_out.diagnostic_note}]"

            evaluated_claims.append(
                ClaimEvaluation(
                    claim_text=claim_text,
                    llm_verdict=llm_out.verdict,
                    llm_confidence=llm_out.confidence,
                    llm_reasoning=reasoning_combined,
                    nli_label=nli_out.label,
                    nli_probabilities=nli_out.probabilities,
                    best_chunk_id=nli_out.best_chunk_id or llm_out.cited_chunk_id,
                    agreement=agreement,
                    disagreement_type=disagreement_type,
                    calibrated_score=calibrated,
                )
            )
            llm_scores.append(llm_raw)
            nli_scores.append(nli_raw)
            calibrated_scores.append(calibrated)

        n = len(evaluated_claims)
        return FaithfulnessResult(
            llm_judge_score=round(sum(llm_scores) / n, 4),
            nli_entailment_score=round(sum(nli_scores) / n, 4),
            cross_checked_score=round(sum(calibrated_scores) / n, 4),
            claims=evaluated_claims,
            disagreement_count=disagreement_count,
            has_judge_disagreement=(disagreement_count > 0),
            contradiction_count=contradiction_count,
            unverifiable_count=unverifiable_count,
            is_abstention=False,
        )
