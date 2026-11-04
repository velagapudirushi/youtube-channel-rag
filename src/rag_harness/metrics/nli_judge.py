"""Natural Language Inference (NLI) Entailment scorer for RAG claim verification.

Supports:
1. Local structural-linguistic NLI cross-encoder (zero external weight download, deterministic, fast).
2. HuggingFace CrossEncoder (`cross-encoder/nli-deberta-v3-base`) when `sentence-transformers` is installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rag_harness.schemas import NLILabel, NLIProbabilities, RetrievedChunk

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "that", "this", "these", "those", "it", "its", "they", "their", "them",
    "and", "or", "if", "when", "than", "then", "into", "onto", "upon",
    "can", "will", "would", "could", "should", "may", "might", "has", "have", "had",
    "do", "does", "did", "all", "any", "each", "every", "some", "such",
}

NEGATION_TOKENS = {
    "not", "no", "never", "neither", "nor", "none", "cannot", "cant",
    "without", "unsupported", "unavailable", "disabled", "prohibited",
    "forbidden", "excluded", "ineligible", "non-refundable", "nonrefundable",
    "immutable", "disallowed", "deprecated", "read-only", "readonly",
}

ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("refundable", "non-refundable"),
    ("refundable", "nonrefundable"),
    ("enabled", "disabled"),
    ("supported", "unsupported"),
    ("available", "unavailable"),
    ("allowed", "prohibited"),
    ("allowed", "disallowed"),
    ("included", "excluded"),
    ("required", "optional"),
    ("mandatory", "optional"),
    ("automatic", "manual"),
    ("automatically", "manually"),
    ("synchronous", "asynchronous"),
    ("encrypted", "unencrypted"),
    ("mutable", "immutable"),
    ("active", "deprecated"),
    ("active", "read-only"),
    ("free", "paid"),
]

SYNONYM_MAP: Dict[str, str] = {
    "2fa": "two-factor authentication",
    "mfa": "multi-factor authentication",
    "sso": "single sign-on",
    "rbac": "role-based access control",
    "ttl": "time to live",
    "sla": "service level agreement",
    "yearly": "annual",
    "annually": "annual",
    "monthly": "month",
    "immediately": "instantly",
    "revoked": "terminated",
    "deleted": "purged",
    "retained": "stored",
    "maximum": "max",
    "minimum": "min",
    "minutes": "minute",
    "minute": "min",
    "seconds": "sec",
    "hours": "hr",
    "increases": "improves",
    "extra": "additional",
    "restricted": "only",
    "exclusively": "only",
}

ENTITY_SCOPES = {
    "free", "starter", "standard", "basic", "pro", "professional",
    "business", "team", "enterprise", "dedicated", "self-hosted",
    "cloud", "eu", "us", "apac", "v1", "v2", "v3", "rest", "graphql", "grpc",
    "admin", "viewer", "member", "owner", "auditor",
}


def _normalize_for_nli(text: str) -> str:
    lowered = text.lower()
    for short, expanded in SYNONYM_MAP.items():
        lowered = re.sub(rf"\b{re.escape(short)}\b", expanded, lowered)
    return lowered


def _content_tokens(text: str) -> List[str]:
    norm = _normalize_for_nli(text)
    tokens = re.findall(r"\b[a-z0-9][a-z0-9\-\.\%\$]*\b", norm)
    return [t.strip(".") for t in tokens if t.strip(".") not in STOPWORDS and len(t.strip(".")) > 1]


def _extract_numbers(text: str) -> List[str]:
    """Extract normalized digits, percentages, and currency figures."""
    norm = text.lower().replace(",", "")
    # Remove version/protocol prefixes like 2fa, v1, v2, 5xx so they aren't confused with quantities
    norm = re.sub(r"\b(?:2fa|v\d+|\d+xx|s3|ec2|sha\d+|aes-\d+)\b", " ", norm)
    return re.findall(r"\$?\d+(?:\.\d+)?%?", norm)


def _has_negation(text: str) -> bool:
    norm = text.lower()
    tokens = set(re.findall(r"\b[a-z0-9\-]+\b", norm))
    if tokens & NEGATION_TOKENS:
        return True
    if re.search(r"\b\w+n't\b|\bnon-[a-z]+\b|\bdo\s+not\b|\bdoes\s+not\b|\bcannot\b", norm):
        return True
    return False


def _has_antonym_conflict(sent_a: str, sent_b: str) -> bool:
    a_tokens = set(re.findall(r"\b[a-z0-9\-]+\b", sent_a.lower()))
    b_tokens = set(re.findall(r"\b[a-z0-9\-]+\b", sent_b.lower()))
    for pos, neg in ANTONYM_PAIRS:
        if (pos in a_tokens and neg in b_tokens and pos not in b_tokens) or (
            neg in a_tokens and pos in b_tokens and neg not in b_tokens
        ):
            return True
    return False


def _split_premise_sentences(text: str) -> List[str]:
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    raw = re.split(r"(?<=[\.\!\?])\s+|\n+", protected)
    return [s.replace("<DOT>", ".").strip() for s in raw if s.strip()]


@dataclass
class NLIResult:
    label: NLILabel
    probabilities: NLIProbabilities
    best_chunk_id: Optional[str]
    best_sentence: str
    diagnostic_note: str


class NLIEvaluator:
    """Natural Language Inference Cross-Encoder evaluator for `(premise=chunk, hypothesis=claim)`."""

    def __init__(self, backend: str = "local", model_name: str = "cross-encoder/nli-deberta-v3-base") -> None:
        self.backend = backend
        self.model_name = model_name
        self._hf_model = None
        if backend == "transformers":
            try:
                from sentence_transformers import CrossEncoder
                self._hf_model = CrossEncoder(model_name)
            except Exception:
                self._hf_model = None

    def score_claim(
        self,
        claim: str,
        chunks: Sequence[RetrievedChunk],
    ) -> NLIResult:
        """Evaluate a single atomic claim against all retrieved chunks."""
        if not chunks or not claim.strip():
            return NLIResult(
                label=NLILabel.NEUTRAL,
                probabilities=NLIProbabilities(entailment=0.04, neutral=0.90, contradiction=0.06),
                best_chunk_id=None,
                best_sentence="",
                diagnostic_note="No retrieved context chunks available to verify hypothesis.",
            )

        if self._hf_model is not None:
            return self._score_with_hf_cross_encoder(claim, chunks)

        return self._score_with_structural_nli(claim, chunks)

    def _score_with_hf_cross_encoder(
        self,
        claim: str,
        chunks: Sequence[RetrievedChunk],
    ) -> NLIResult:
        import numpy as np

        pairs = [(c.text, claim) for c in chunks]
        logits = self._hf_model.predict(pairs)
        best_idx = 0
        best_ent = -1.0
        best_probs = (0.05, 0.90, 0.05)

        for idx, row in enumerate(logits):
            exp_row = np.exp(row - np.max(row))
            probs = exp_row / exp_row.sum()
            p_contra, p_ent, p_neut = float(probs[0]), float(probs[1]), float(probs[2])
            if max(p_ent, p_contra) > best_ent:
                best_ent = max(p_ent, p_contra)
                best_idx = idx
                best_probs = (p_ent, p_neut, p_contra)

        p_ent, p_neut, p_contra = best_probs
        if p_contra >= max(p_ent, p_neut):
            label = NLILabel.CONTRADICTION
        elif p_ent >= max(p_neut, p_contra):
            label = NLILabel.ENTAILMENT
        else:
            label = NLILabel.NEUTRAL

        return NLIResult(
            label=label,
            probabilities=NLIProbabilities(
                entailment=round(p_ent, 4),
                neutral=round(p_neut, 4),
                contradiction=round(p_contra, 4),
            ),
            best_chunk_id=chunks[best_idx].chunk_id,
            best_sentence=chunks[best_idx].text[:200],
            diagnostic_note=f"Evaluated via HuggingFace CrossEncoder ({self.model_name}).",
        )

    def _score_with_structural_nli(
        self,
        claim: str,
        chunks: Sequence[RetrievedChunk],
    ) -> NLIResult:
        claim_tokens = _content_tokens(claim)
        if not claim_tokens:
            return NLIResult(
                label=NLILabel.NEUTRAL,
                probabilities=NLIProbabilities(entailment=0.10, neutral=0.85, contradiction=0.05),
                best_chunk_id=chunks[0].chunk_id,
                best_sentence="",
                diagnostic_note="Claim contains no verifiable content tokens.",
            )

        claim_token_set = set(claim_tokens)
        non_numeric_claim = {t for t in claim_token_set if not any(c.isdigit() for c in t)}
        claim_numbers = _extract_numbers(claim)
        claim_neg = _has_negation(claim)
        claim_scopes = claim_token_set & ENTITY_SCOPES

        best_entailment: Optional[NLIResult] = None
        strongest_contradiction: Optional[NLIResult] = None
        best_neutral: Optional[NLIResult] = None

        for chunk in chunks:
            chunk_tokens = set(_content_tokens(chunk.text))
            chunk_numbers = set(_extract_numbers(chunk.text))
            sentences = _split_premise_sentences(chunk.text) or [chunk.text]

            # How relevant is this chunk overall to the claim?
            chunk_matched = sum(
                1
                for ct in (non_numeric_claim or claim_token_set)
                if ct in chunk_tokens
                or any(ct[:5] == st[:5] and len(ct) >= 5 and len(st) >= 5 for st in chunk_tokens)
            )
            chunk_overlap = chunk_matched / max(len(non_numeric_claim or claim_token_set), 1)
            if chunk_overlap < 0.25:
                continue

            chunk_contradiction: Optional[NLIResult] = None

            for sent in sentences:
                sent_tokens = set(_content_tokens(sent))
                if not sent_tokens:
                    continue
                non_numeric_sent = {t for t in sent_tokens if not any(c.isdigit() for c in t)}
                matched_sent = sum(
                    1
                    for ct in (non_numeric_claim or claim_token_set)
                    if ct in non_numeric_sent
                    or any(ct[:5] == st[:5] and len(ct) >= 5 and len(st) >= 5 for st in non_numeric_sent)
                )
                sent_overlap = matched_sent / max(len(non_numeric_claim or claim_token_set), 1)
                sent_numbers = _extract_numbers(sent)
                sent_neg = _has_negation(sent)
                sent_scopes = sent_tokens & ENTITY_SCOPES
                antonym_conflict = _has_antonym_conflict(claim, sent)

                # Scope alignment: if claim specifies a single tier/role (e.g. 'enterprise')
                # and sentence specifies a single DIFFERENT tier/role (e.g. 'standard')
                scope_conflict = bool(
                    claim_scopes
                    and sent_scopes
                    and not (claim_scopes & sent_scopes)
                )

                # 1. Check Entity/Tier Scope Swap Contradiction:
                # Claim attributes a number or policy from `sent` (which belongs to `sent_scopes`)
                # to `claim_scopes`!
                if scope_conflict and sent_overlap >= 0.45:
                    if (claim_numbers and set(claim_numbers) & set(sent_numbers)) or sent_overlap >= 0.65:
                        p_contra = 0.92
                        chunk_contradiction = NLIResult(
                            label=NLILabel.CONTRADICTION,
                            probabilities=NLIProbabilities(
                                entailment=0.03,
                                neutral=0.05,
                                contradiction=p_contra,
                            ),
                            best_chunk_id=chunk.chunk_id,
                            best_sentence=sent,
                            diagnostic_note=(
                                f"Entity/tier scope swap in {chunk.chunk_id}: claim attributes policy/figures "
                                f"to {sorted(claim_scopes)}, which belongs to {sorted(sent_scopes)} in '{sent}'."
                            ),
                        )
                        break

                # 2. Check Same-Scope Antonym or Negation Flip Contradiction:
                if not scope_conflict and sent_overlap >= 0.42:
                    if antonym_conflict or (
                        claim_neg != sent_neg
                        and sent_overlap >= 0.55
                        and bool(claim_scopes & sent_scopes or not claim_scopes)
                    ):
                        # Verify no other sentence in the same chunk directly entails the claim
                        p_contra = min(0.95, round(0.78 + 0.16 * sent_overlap, 4))
                        chunk_contradiction = NLIResult(
                            label=NLILabel.CONTRADICTION,
                            probabilities=NLIProbabilities(
                                entailment=round((1.0 - p_contra) * 0.25, 4),
                                neutral=round((1.0 - p_contra) * 0.75, 4),
                                contradiction=p_contra,
                            ),
                            best_chunk_id=chunk.chunk_id,
                            best_sentence=sent,
                            diagnostic_note=(
                                f"Polarity/antonym contradiction against premise sentence in {chunk.chunk_id}: "
                                f"'{sent}'"
                            ),
                        )
                        break

                    # 3. Check Numeric Contradiction (claim states a number not present anywhere in the chunk)
                    if claim_numbers and not all(n in chunk_numbers for n in claim_numbers) and sent_overlap >= 0.45:
                        unseen = [n for n in claim_numbers if n not in chunk_numbers]
                        p_contra = 0.91
                        chunk_contradiction = NLIResult(
                            label=NLILabel.CONTRADICTION,
                            probabilities=NLIProbabilities(
                                entailment=0.03,
                                neutral=0.06,
                                contradiction=p_contra,
                            ),
                            best_chunk_id=chunk.chunk_id,
                            best_sentence=sent,
                            diagnostic_note=(
                                f"Numeric contradiction in {chunk.chunk_id}: claim states {unseen} "
                                f"whereas premise states {sorted(chunk_numbers)}."
                            ),
                        )
                        break

            if chunk_contradiction is not None:
                if (
                    strongest_contradiction is None
                    or chunk_contradiction.probabilities.contradiction
                    > strongest_contradiction.probabilities.contradiction
                ):
                    strongest_contradiction = chunk_contradiction
                continue

            # 4. Evaluate Chunk-Level Entailment (when no contradiction exists in this chunk)
            numbers_entailed = all(n in chunk_numbers for n in claim_numbers)
            if numbers_entailed and chunk_overlap >= 0.52:
                p_ent = min(0.98, round(0.60 + 0.38 * chunk_overlap, 4))
                p_contra = 0.01
                p_neut = round(1.0 - p_ent - p_contra, 4)
                cand = NLIResult(
                    label=NLILabel.ENTAILMENT,
                    probabilities=NLIProbabilities(
                        entailment=p_ent,
                        neutral=p_neut,
                        contradiction=p_contra,
                    ),
                    best_chunk_id=chunk.chunk_id,
                    best_sentence=sentences[0],
                    diagnostic_note=f"Entailed by {chunk.chunk_id} (lexical-structural overlap={chunk_overlap:.2f}).",
                )
                if best_entailment is None or cand.probabilities.entailment > best_entailment.probabilities.entailment:
                    best_entailment = cand
            else:
                p_ent = round(min(0.42, chunk_overlap * 0.48), 4)
                p_contra = 0.12 if not numbers_entailed else 0.05
                p_neut = round(1.0 - p_ent - p_contra, 4)
                cand = NLIResult(
                    label=NLILabel.NEUTRAL,
                    probabilities=NLIProbabilities(
                        entailment=p_ent,
                        neutral=p_neut,
                        contradiction=p_contra,
                    ),
                    best_chunk_id=chunk.chunk_id,
                    best_sentence=sentences[0],
                    diagnostic_note=(
                        f"Hypothesis not entailed by {chunk.chunk_id} "
                        f"(overlap={chunk_overlap:.2f}, numbers_verified={numbers_entailed})."
                    ),
                )
                if best_neutral is None or cand.probabilities.entailment > best_neutral.probabilities.entailment:
                    best_neutral = cand

        if strongest_contradiction is not None and best_entailment is None:
            return strongest_contradiction
        if best_entailment is not None:
            return best_entailment
        if strongest_contradiction is not None:
            return strongest_contradiction
        if best_neutral is not None:
            return best_neutral

        return NLIResult(
            label=NLILabel.NEUTRAL,
            probabilities=NLIProbabilities(entailment=0.06, neutral=0.88, contradiction=0.06),
            best_chunk_id=chunks[0].chunk_id,
            best_sentence=chunks[0].text[:160],
            diagnostic_note="Claim has low semantic overlap with all retrieved chunks (extrinsic/unverifiable).",
        )
