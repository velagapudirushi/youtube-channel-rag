"""Benchmark knowledge base corpus and 15-query evaluation dataset for public RAG repo demos."""

from __future__ import annotations

from typing import Dict, List
from rag_harness.schemas import QuestionItem

KNOWLEDGE_CORPUS: List[Dict[str, str]] = [
    {
        "chunk_id": "doc_01_refund_policy",
        "title": "Subscription Billing & Refund Policy",
        "text": (
            "Standard tier subscriptions include a 30-day full refund window from the initial purchase date. "
            "Pro tier subscriptions are eligible for a prorated refund within 14 days. "
            "Enterprise tier annual contracts are strictly non-refundable once provisioned, though unused credits "
            "can be reallocated across workspaces."
        ),
    },
    {
        "chunk_id": "doc_02_rate_limits",
        "title": "API Rate Limits & Throttling",
        "text": (
            "Public REST API endpoints enforce a default rate limit of 120 requests per minute for Standard plans "
            "and 600 requests per minute for Pro plans. Enterprise workspaces support 5,000 requests per minute "
            "with burst capacity up to 8,000 requests. Exceeding the limit returns HTTP status 429 Too Many Requests "
            "with a Retry-After header."
        ),
    },
    {
        "chunk_id": "doc_03_cli_2fa_reset",
        "title": "CLI Administration & 2FA Recovery",
        "text": (
            "Organization owners can reset two-factor authentication (2FA) for a locked-out member using the "
            "admin CLI command `vaultctl auth mfa-reset --user <email> --reason-code SEC-402`. "
            "This command revokes all active session tokens immediately and sends a one-time hardware key enrollment link."
        ),
    },
    {
        "chunk_id": "doc_04_encryption_kms",
        "title": "Data Encryption at Rest and in Transit",
        "text": (
            "All customer vector indexes and document blobs are encrypted at rest using AES-256-GCM with keys "
            "rotated automatically every 90 days. Data in transit requires TLS 1.3; TLS 1.1 and TLS 1.2 connections "
            "are rejected at the edge proxy. Customer-Managed Keys (CMK) via AWS KMS are supported only on Enterprise plans."
        ),
    },
    {
        "chunk_id": "doc_05_webhook_retries",
        "title": "Webhook Delivery & Exponential Backoff",
        "text": (
            "When a customer webhook endpoint returns a 5xx server error or times out after 10 seconds, the event "
            "dispatcher retries delivery up to 6 times using exponential backoff starting at 30 seconds. "
            "After 6 consecutive failures over 8 hours, the webhook endpoint is automatically disabled and marked unhealthy."
        ),
    },
    {
        "chunk_id": "doc_06_data_retention",
        "title": "Audit Logs & Zero Data Retention (ZDR)",
        "text": (
            "By default, inference trace logs are retained for 30 days on Standard and Pro tiers for debugging. "
            "Audit security logs are retained for 365 days. Zero Data Retention (ZDR)—where prompts and completions "
            "are never written to disk—is disabled by default and requires signing an Enterprise BAA addendum."
        ),
    },
    {
        "chunk_id": "doc_07_sso_saml",
        "title": "Single Sign-On (SSO) & SCIM Provisioning",
        "text": (
            "SAML 2.0 and OIDC Single Sign-On (SSO) are available on Pro and Enterprise plans. "
            "Automated user provisioning and deprovisioning via SCIM 2.0 is restricted exclusively to Enterprise plans "
            "and syncs group membership every 15 minutes."
        ),
    },
    {
        "chunk_id": "doc_08_vector_index_hnsw",
        "title": "Vector Index HNSW Tuning Parameters",
        "text": (
            "The default HNSW vector index uses `m=16` and `ef_construction=200` with cosine distance. "
            "At query time, the default search depth is `ef_search=64`. Increasing `ef_search` to 128 improves "
            "Recall@10 by approximately 4.2% at the cost of 18ms additional p99 query latency."
        ),
    },
    {
        "chunk_id": "doc_09_error_err_idx_809",
        "title": "Troubleshooting Error Code ERR_IDX_809",
        "text": (
            "Error code `ERR_IDX_809` occurs when an ingestion batch contains embedding vectors whose dimension "
            "does not match the collection schema (for example, sending 1536-dim vectors to a 768-dim index). "
            "Resolve this by running `vaultctl collection migrate --reembed` or specifying `--dim 1536` at creation."
        ),
    },
    {
        "chunk_id": "doc_10_eu_data_residency",
        "title": "Regional Data Residency & Cross-Border Replication",
        "text": (
            "Workspaces pinned to the `eu-central-1` (Frankfurt) region store all primary shards and daily snapshots "
            "strictly within Frankfurt availability zones. Cross-region replication outside the EU is prohibited "
            "and cannot be enabled even by organization administrators."
        ),
    },
    {
        "chunk_id": "doc_11_sla_credits",
        "title": "Uptime SLA & Service Credit Schedule",
        "text": (
            "Enterprise workspaces carry a 99.99% monthly uptime SLA. If Enterprise monthly uptime falls between "
            "99.90% and 99.99%, Enterprise customers receive a 10% service credit. If Enterprise uptime drops below "
            "99.0%, customers receive a 30% service credit. Standard tier workspaces do not carry a financially backed uptime SLA."
        ),
    },
    {
        "chunk_id": "doc_12_backup_rpo_rto",
        "title": "Disaster Recovery, RPO, and RTO Targets",
        "text": (
            "Production database clusters take continuous WAL snapshots with a Recovery Point Objective (RPO) of "
            "5 minutes and a Recovery Time Objective (RTO) of 30 minutes for regional failover. "
            "Point-in-time recovery (PITR) allows restoring to any second within the last 14 days."
        ),
    },
    {
        "chunk_id": "doc_13_helm_gpu_flag",
        "title": "Self-Hosted Kubernetes Helm Deployment",
        "text": (
            "To deploy the local NLI reranker pod on GPU nodes in Kubernetes, pass "
            "`--set reranker.gpu.enabled=true --set reranker.resources.vramMinGb=8` during `helm upgrade --install`. "
            "Minimum supported NVIDIA driver version is 535.104 with CUDA 12.1."
        ),
    },
    {
        "chunk_id": "doc_14_graphql_deprecation",
        "title": "GraphQL API v1 Deprecation Notice",
        "text": (
            "The legacy `/v1/graphql` endpoint is deprecated and read-only as of Q2. All mutation operations "
            "must use `/v2/rest/mutations` or the gRPC ingestion stream on port 9443. Queries to `/v1/graphql` "
            "will be permanently sunset on December 31."
        ),
    },
    {
        "chunk_id": "doc_15_rbac_roles",
        "title": "Workspace RBAC Roles & Permissions",
        "text": (
            "The `Auditor` role grants read-only access to billing invoices, security logs, and evaluation reports "
            "but cannot view raw document chunks or execute API queries. The `Member` role can run queries and "
            "upload documents, while only `Owner` and `Admin` roles can delete collections or rotate API keys."
        ),
    },
    # Distractor chunks that share high vocabulary overlap with the core chunks
    {
        "chunk_id": "doc_16_distractor_billing_faq",
        "title": "General Billing FAQ & Invoice Cycles",
        "text": (
            "Monthly invoices are generated on the 1st of each calendar month. Customers can pay via ACH, wire transfer, "
            "or credit card. Late payments accrue a 1.5% monthly fee after a 30-day grace period."
        ),
    },
    {
        "chunk_id": "doc_17_distractor_cli_install",
        "title": "Installing the vaultctl CLI Tool",
        "text": (
            "Install `vaultctl` on macOS using `brew install vaultctl` or via pip with `pip install vaultctl-cli`. "
            "Authenticate your local terminal session by running `vaultctl login --sso`."
        ),
    },
    {
        "chunk_id": "doc_18_distractor_sdk_retries",
        "title": "Python SDK Client Configuration",
        "text": (
            "The Python SDK client defaults to a 60-second request timeout and 3 automatic retries for network "
            "connection resets. Configure custom timeouts via `Client(timeout=30.0, max_retries=5)`."
        ),
    },
]


BENCHMARK_QUESTIONS: List[QuestionItem] = [
    QuestionItem(
        query_id="q_01",
        question="What is the refund window for Enterprise tier annual contracts?",
        ground_truth_answer=(
            "Enterprise tier annual contracts are strictly non-refundable once provisioned, "
            "though unused credits can be reallocated across workspaces."
        ),
        golden_chunk_ids=["doc_01_refund_policy"],
        key_facts=["non-refundable", "Enterprise", "unused credits"],
        category="policy-tier-trap",
    ),
    QuestionItem(
        query_id="q_02",
        question="What are the API rate limits for Standard, Pro, and Enterprise plans?",
        ground_truth_answer=(
            "Standard plans allow 120 requests per minute, Pro plans allow 600 requests per minute, "
            "and Enterprise workspaces support 5,000 requests per minute (with burst up to 8,000)."
        ),
        golden_chunk_ids=["doc_02_rate_limits"],
        key_facts=["120 requests per minute", "600 requests per minute", "5,000 requests per minute", "429"],
        category="numeric-precision",
    ),
    QuestionItem(
        query_id="q_03",
        question="What exact CLI command and reason code resets 2FA for a locked-out user?",
        ground_truth_answer=(
            "Run `vaultctl auth mfa-reset --user <email> --reason-code SEC-402`, which immediately revokes "
            "active session tokens and sends a one-time hardware key enrollment link."
        ),
        golden_chunk_ids=["doc_03_cli_2fa_reset"],
        key_facts=["vaultctl auth mfa-reset", "SEC-402", "revokes"],
        category="exact-cli-command",
    ),
    QuestionItem(
        query_id="q_04",
        question="How is data encrypted at rest and in transit, and how often are keys rotated?",
        ground_truth_answer=(
            "Data at rest is encrypted using AES-256-GCM with keys rotated automatically every 90 days. "
            "Data in transit requires TLS 1.3, and Customer-Managed Keys (CMK) via AWS KMS require an Enterprise plan."
        ),
        golden_chunk_ids=["doc_04_encryption_kms"],
        key_facts=["AES-256-GCM", "90 days", "TLS 1.3", "Enterprise"],
        category="security-compliance",
    ),
    QuestionItem(
        query_id="q_05",
        question="How many times does the webhook dispatcher retry on a 5xx error and when is the webhook disabled?",
        ground_truth_answer=(
            "It retries up to 6 times using exponential backoff starting at 30 seconds when an endpoint returns 5xx "
            "or times out after 10 seconds. After 6 consecutive failures over 8 hours, the webhook is disabled."
        ),
        golden_chunk_ids=["doc_05_webhook_retries"],
        key_facts=["6 times", "30 seconds", "10 seconds", "8 hours", "disabled"],
        category="numeric-precision",
    ),
    QuestionItem(
        query_id="q_06",
        question="Is Zero Data Retention (ZDR) enabled by default on Standard and Pro tiers?",
        ground_truth_answer=(
            "No, Zero Data Retention (ZDR) is disabled by default and requires signing an Enterprise BAA addendum. "
            "Standard and Pro tiers retain inference trace logs for 30 days by default."
        ),
        golden_chunk_ids=["doc_06_data_retention"],
        key_facts=["disabled by default", "Enterprise BAA", "30 days"],
        category="negation-polarity",
    ),
    QuestionItem(
        query_id="q_07",
        question="Which plans support SAML SSO versus automated SCIM 2.0 user provisioning?",
        ground_truth_answer=(
            "SAML 2.0 and OIDC SSO are supported on both Pro and Enterprise plans, whereas automated SCIM 2.0 "
            "provisioning is restricted exclusively to Enterprise plans and syncs every 15 minutes."
        ),
        golden_chunk_ids=["doc_07_sso_saml"],
        key_facts=["Pro and Enterprise", "SCIM 2.0", "exclusively to Enterprise", "15 minutes"],
        category="policy-tier-trap",
    ),
    QuestionItem(
        query_id="q_08",
        question="What are the default HNSW index parameters and what happens when ef_search is raised to 128?",
        ground_truth_answer=(
            "The default HNSW index uses m=16, ef_construction=200, and ef_search=64. Raising ef_search to 128 "
            "increases Recall@10 by about 4.2% with 18ms extra p99 latency."
        ),
        golden_chunk_ids=["doc_08_vector_index_hnsw"],
        key_facts=["m=16", "ef_construction=200", "ef_search=64", "4.2%", "18ms"],
        category="numeric-precision",
    ),
    QuestionItem(
        query_id="q_09",
        question="What causes error code ERR_IDX_809 and how do you fix it?",
        ground_truth_answer=(
            "ERR_IDX_809 is caused by ingesting embedding vectors whose dimension does not match the collection schema "
            "(e.g., 1536-dim into a 768-dim index). Fix it via `vaultctl collection migrate --reembed` or `--dim 1536`."
        ),
        golden_chunk_ids=["doc_09_error_err_idx_809"],
        key_facts=["ERR_IDX_809", "dimension", "vaultctl collection migrate --reembed"],
        category="exact-error-code",
    ),
    QuestionItem(
        query_id="q_10",
        question="Can an organization admin enable cross-region replication outside the EU for an eu-central-1 workspace?",
        ground_truth_answer=(
            "No. Workspaces pinned to eu-central-1 store all primary shards and snapshots strictly in Frankfurt, "
            "and cross-region replication outside the EU is prohibited even for organization administrators."
        ),
        golden_chunk_ids=["doc_10_eu_data_residency"],
        key_facts=["prohibited", "eu-central-1", "Frankfurt"],
        category="negation-polarity",
    ),
    QuestionItem(
        query_id="q_11",
        question="What uptime SLA and service credit does a Standard tier workspace receive if uptime drops below 99.0%?",
        ground_truth_answer=(
            "Standard tier workspaces do not carry a financially backed uptime SLA and receive no service credits. "
            "The 99.99% SLA and 10%/30% service credits apply only to Enterprise workspaces."
        ),
        golden_chunk_ids=["doc_11_sla_credits"],
        key_facts=["do not carry", "Standard", "Enterprise", "99.99%"],
        category="policy-tier-trap",
    ),
    QuestionItem(
        query_id="q_12",
        question="What are the RPO, RTO, and Point-in-Time Recovery (PITR) windows for production database clusters?",
        ground_truth_answer=(
            "Production clusters have a 5-minute Recovery Point Objective (RPO), a 30-minute Recovery Time Objective (RTO), "
            "and Point-in-Time Recovery (PITR) to any second within the last 14 days."
        ),
        golden_chunk_ids=["doc_12_backup_rpo_rto"],
        key_facts=["5 minutes", "30 minutes", "14 days"],
        category="numeric-precision",
    ),
    QuestionItem(
        query_id="q_13",
        question="What Helm flags and CUDA/driver versions are needed to enable GPU for the self-hosted reranker pod?",
        ground_truth_answer=(
            "Pass `--set reranker.gpu.enabled=true --set reranker.resources.vramMinGb=8` with `helm upgrade --install`. "
            "It requires NVIDIA driver 535.104+ and CUDA 12.1."
        ),
        golden_chunk_ids=["doc_13_helm_gpu_flag"],
        key_facts=["reranker.gpu.enabled=true", "vramMinGb=8", "535.104", "CUDA 12.1"],
        category="exact-cli-command",
    ),
    QuestionItem(
        query_id="q_14",
        question="Is the /v1/graphql endpoint still accepting mutations, and what port does gRPC ingestion use?",
        ground_truth_answer=(
            "No, `/v1/graphql` is deprecated and read-only as of Q2 (sunset December 31). Mutations must use "
            "`/v2/rest/mutations` or the gRPC ingestion stream on port 9443."
        ),
        golden_chunk_ids=["doc_14_graphql_deprecation"],
        key_facts=["read-only", "/v2/rest/mutations", "9443", "December 31"],
        category="negation-polarity",
    ),
    QuestionItem(
        query_id="q_15",
        question="Can a user with the Auditor RBAC role view raw document chunks or delete a collection?",
        ground_truth_answer=(
            "No. The Auditor role has read-only access to billing invoices, security logs, and evaluation reports, "
            "and cannot view raw document chunks, execute API queries, or delete collections."
        ),
        golden_chunk_ids=["doc_15_rbac_roles"],
        key_facts=["cannot view raw document chunks", "read-only", "delete collections"],
        category="negation-polarity",
    ),
]
