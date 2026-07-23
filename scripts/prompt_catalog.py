"""Diverse prompt templates for benchmark trace generation."""

from __future__ import annotations

import random
from typing import Callable

PRODUCTS = [
    "Orbit Analytics",
    "Nimbus CRM",
    "PulsePay",
    "Vertex Shield",
    "Harbor Logistics",
    "Lumen Docs",
    "Forge DevOps",
    "Cascade BI",
]

CUSTOMERS = [
    "Northwind Traders",
    "BluePeak Health",
    "Summit Retail",
    "Helios Manufacturing",
    "Cedar Financial",
    "Atlas Airlines",
    "Nova Education",
    "Ironclad Insurance",
]

TEAMS = [
    "platform engineering",
    "customer success",
    "sales operations",
    "security",
    "data science",
    "product marketing",
    "finance",
    "legal",
]

REGIONS = ["us-central1", "us-east1", "europe-west4", "asia-southeast1"]

LANGUAGES = ["Python", "Go", "TypeScript", "Java", "Rust", "SQL"]

ERROR_TYPES = [
    "timeout",
    "OOM",
    "permission denied",
    "schema mismatch",
    "rate limit exceeded",
    "stale cache",
    "network partition",
]


def _pick(rng: random.Random, items: list[str]) -> str:
    return rng.choice(items)


def _paragraphs(rng: random.Random, count: int) -> str:
    topics = [
        "latency increased after the last deploy",
        "error budget consumption spiked overnight",
        "support tickets mention duplicate charges",
        "the dashboard shows inconsistent totals",
        "a partner API changed response fields",
        "on-call noticed elevated 503 responses",
        "finance flagged a revenue recognition gap",
        "compliance asked for an audit trail export",
    ]
    lines = []
    for i in range(count):
        topic = rng.choice(topics)
        metric = rng.randint(12, 98)
        lines.append(
            f"Observation {i + 1}: {topic}; severity {rng.choice(['low', 'medium', 'high'])}; "
            f"impact score {metric}; owner {_pick(rng, TEAMS)}."
        )
    return "\n".join(lines)


def template_code_review(rng: random.Random, req_id: int) -> str:
    lang = _pick(rng, LANGUAGES)
    product = _pick(rng, PRODUCTS)
    return f"""Review this {lang} change from PR #{4200 + req_id} for {product}.

```{lang.lower()}
def reconcile_usage(account_id, window_hours=24):
    rows = fetch_events(account_id)  # may return None on cache miss
    total = sum(r['units'] for r in rows)
    if total > LIMIT:
        emit_alert(account_id, total)
    return total
```

The author claims this fixes miscounted usage in { _pick(rng, REGIONS) }.
Questions:
1. What failure modes remain?
2. Suggest tests and a safer implementation.
3. Estimate operational risk if deployed Friday 5pm.

Context from the ticket: {_paragraphs(rng, 4)}"""


def template_incident_triage(rng: random.Random, req_id: int) -> str:
    product = _pick(rng, PRODUCTS)
    err = _pick(rng, ERROR_TYPES)
    return f"""You are on-call for {product}. Incident INC-{9000 + req_id} opened {rng.randint(5, 45)} minutes ago.

Symptoms:
- p95 latency {rng.randint(800, 2400)}ms (SLO 400ms)
- {rng.randint(2, 18)}% of requests failing with {err}
- Primary cluster: {_pick(rng, REGIONS)}

Recent changes:
- Config rollout v{rng.randint(1, 9)}.{rng.randint(0, 20)} at {rng.randint(10, 23)}:{rng.randint(10, 59)} UTC
- Autoscaler max raised from {rng.randint(20, 40)} to {rng.randint(41, 80)}

Log excerpt:
```
[{req_id}] WARN handler timeout after {rng.randint(30, 120)}s upstream={ _pick(rng, CUSTOMERS) }
[{req_id}] ERROR retry exhausted path=/v1/inference status=503
[{req_id}] INFO cache hit ratio={rng.uniform(0.4, 0.9):.2f}
```

Provide: likely root cause (ranked), immediate mitigation steps, and what to communicate to { _pick(rng, CUSTOMERS) }."""


def template_customer_email(rng: random.Random, req_id: int) -> str:
    customer = _pick(rng, CUSTOMERS)
    product = _pick(rng, PRODUCTS)
    return f"""Draft a professional reply to this customer email about {product}.

From: ops@{customer.lower().replace(' ', '')}.com
Subject: Billing discrepancy on invoice INV-{30000 + req_id}

Hello team,

We were charged ${rng.randint(1200, 9800)} for API usage last month but our internal meter shows
${rng.randint(800, 7000)}. Three applications ({rng.choice(['ETL', 'chatbot', 'search', 'mobile app'])},
{rng.choice(['reporting', 'fraud checks', 'recommendations'])}, and
{rng.choice(['archival', 'real-time scoring'])}) share one API key.

Please explain the delta, provide a line-item breakdown, and confirm whether rate limits changed in
{_pick(rng, REGIONS)}. We need a response before our accounts payable cutoff on Friday.

Regards,
{rng.choice(['Jordan', 'Priya', 'Marcus', 'Elena', 'Sam'])} {rng.choice(['Okonkwo', 'Chen', 'Rivera', 'Berg', 'Patel'])}

Write the reply: acknowledge issue, request any missing info, propose next steps, and keep tone calm."""


def template_sql_analyst(rng: random.Random, req_id: int) -> str:
    product = _pick(rng, PRODUCTS)
    return f"""You are a data analyst for {product}. Write SQL (BigQuery dialect) and explain your approach.

Business question Q-{req_id}:
How many enterprise customers had more than {rng.randint(3, 12)} support escalations AND monthly active usage
above the {rng.randint(70, 95)}th percentile in the last {rng.randint(28, 90)} days?

Available tables:
- accounts(account_id, segment, created_at, region)
- usage_daily(account_id, date, api_calls, tokens, cost_usd)
- support_tickets(ticket_id, account_id, opened_at, priority, escalated bool)

Also list two caveats about data quality and one chart you'd show to {_pick(rng, TEAMS)}.

Supplemental notes:
{_paragraphs(rng, 5)}"""


def template_policy_qa(rng: random.Random, req_id: int) -> str:
    return f"""Answer this internal policy question for employees (ID POL-{req_id}).

Question from {_pick(rng, TEAMS)}:
Can we store customer prompt logs from { _pick(rng, PRODUCTS) } in {_pick(rng, REGIONS)} for
{rng.randint(14, 365)} days to fine-tune an internal model?

Constraints mentioned:
- Contract with { _pick(rng, CUSTOMERS) } prohibits training on their data without written consent.
- GDPR/CCPA deletion requests must complete within {rng.randint(7, 30)} days.
- Security requires encryption at rest and audit logging.

Provide: clear yes/no/conditional answer, required approvals, and a short checklist for engineers."""


def template_roadmap(rng: random.Random, req_id: int) -> str:
    product = _pick(rng, PRODUCTS)
    return f"""Prioritize these {product} roadmap items for Q{rng.randint(1, 4)} (request RP-{req_id}).

Candidates:
A) Reduce cold-start latency by {rng.randint(20, 50)}% for models under 8B params
B) Self-serve SSO for {rng.randint(5, 40)} enterprise tenants waiting on SCIM
C) Usage-based billing export API (requested by { _pick(rng, CUSTOMERS) })
D) Multi-region failover for {_pick(rng, REGIONS)}
E) Admin audit log UI for compliance reviews

Team capacity: {rng.randint(2, 5)} engineers for 6 weeks. Dependencies: item A helps C; B blocks two deals.

Deliver: ranked list with rationale, risks, and what to defer. Use markdown headings."""


def template_runbook(rng: random.Random, req_id: int) -> str:
    return f"""Improve this runbook section (RB-{req_id}) for {_pick(rng, TEAMS)}.

Current draft:
1. Check dashboard
2. Restart service if red
3. Escalate

Environment: GKE cluster serving { _pick(rng, PRODUCTS) } in {_pick(rng, REGIONS)}.
Failure mode observed: { _pick(rng, ERROR_TYPES) } during traffic spike to {rng.randint(200, 5000)} RPS.

Rewrite into a actionable runbook with verification commands, rollback criteria, communication template,
and estimated time boxes. Assume reader is mid-level engineer on first on-call rotation.

Background:
{_paragraphs(rng, 4)}"""


def template_api_design(rng: random.Random, req_id: int) -> str:
    product = _pick(rng, PRODUCTS)
    return f"""Review this API proposal for {product} (API-{req_id}).

Endpoint: POST /v2/inference/batch
Body: {{ "model": "...", "requests": [{{"prompt": "...", "max_tokens": 128}}], "priority": "normal|high" }}
Response: 202 Accepted + job_id, or 429 if queue depth > {rng.randint(100, 2000)}

Open questions from { _pick(rng, CUSTOMERS) }:
- Idempotency for retried batch uploads
- Webhook vs polling for completion
- Token accounting when partial batch fails

Critique the design: naming, error model, backwards compatibility, rate limits, and security.
Suggest concrete JSON examples for success and two error cases."""


def template_meeting_notes(rng: random.Random, req_id: int) -> str:
    return f"""Summarize these meeting notes and extract action items (MTG-{req_id}).

Attendees: {_pick(rng, TEAMS)}, {_pick(rng, TEAMS)}, PM, EM
Topic: Launch readiness for { _pick(rng, PRODUCTS) } integration with { _pick(rng, CUSTOMERS) }

Notes:
- Demo on Tuesday; need stable p95 < {rng.randint(300, 800)}ms
- Legal reviewing DPA section 4.{rng.randint(1, 9)} (data residency: {_pick(rng, REGIONS)})
- QA found {rng.randint(3, 15)} bugs; {rng.randint(1, 5)} marked release blockers
- Marketing wants public case study quotes by month end
- Open debate: ship with feature flag at {rng.randint(5, 25)}% or delay one week

Output: executive summary (3 bullets), action items table (owner, due date), and risks."""


def template_docs_howto(rng: random.Random, req_id: int) -> str:
    product = _pick(rng, PRODUCTS)
    return f"""Write a how-to doc section for developers integrating {product} (DOC-{req_id}).

Cover:
- Authenticating with short-lived tokens (not long-lived API keys)
- Sending a streaming chat completion with max {rng.randint(256, 2048)} output tokens
- Handling 429/503 with exponential backoff
- Pinning model version for reproducible evals

Audience: backend engineers new to the platform. Include one minimal curl example and a short Python snippet.
Mention common mistake: reusing the same request id across retries.

Extra context from support:
{_paragraphs(rng, 3)}"""


def template_finance_forecast(rng: random.Random, req_id: int) -> str:
    return f"""Build a concise finance narrative for forecast FC-{req_id}.

Inputs:
- Monthly recurring revenue: ${rng.randint(800, 950)}K, growth {rng.uniform(1, 8):.1f}%
- Inference COGS moved from ${rng.randint(40, 90)}K to ${rng.randint(91, 180)}K after { _pick(rng, REGIONS) } expansion
- Sales pipeline: {rng.randint(6, 22)} enterprise deals, median ACV ${rng.randint(80, 350)}K
- Churn: {rng.uniform(0.5, 3.5):.1f}% logo churn; expansion in { _pick(rng, CUSTOMERS) }

CFO asks: Where will gross margin be in two quarters if usage grows {rng.randint(15, 60)}% but we migrate
{rng.randint(20, 70)}% of workloads from GPU to TPU?

Respond with assumptions, scenario table (bear/base/bull), and two levers to protect margin."""


def template_security_review(rng: random.Random, req_id: int) -> str:
    product = _pick(rng, PRODUCTS)
    return f"""Perform a lightweight security review of this design (SEC-{req_id}).

Service: {product} admin console
Changes:
- Adds OAuth login with external IdP
- Stores session tokens in Redis for {rng.randint(1, 24)} hours
- New endpoint exports all tenant configs as JSON (admin role)

Threat model notes:
- Shared Redis cluster also used for job queues
- Admin role granted to {rng.randint(50, 400)} customer users
- Logs currently include raw Authorization headers

List top 5 risks, recommended controls, and what to verify before pen test."""


def template_comparison(rng: random.Random, req_id: int) -> str:
    return f"""Compare deployment options for inference workload WL-{req_id}.

Option 1: GKE with {rng.choice(['L4', 'A100', 'H100'])} GPU node pool in {_pick(rng, REGIONS)}
Option 2: GKE with TPU v{rng.choice(['5e', '6e'])} slice, same model ({ _pick(rng, PRODUCTS) })
Option 3: Managed API vendor (unknown unit economics)

Evaluation criteria: cost at {rng.randint(50, 500)} RPS, ops burden, tail latency, migration effort.
Use the notes below and produce a decision matrix plus recommendation for a cost-conscious team.

Notes:
{_paragraphs(rng, 6)}"""


TEMPLATES: list[Callable[[random.Random, int], str]] = [
    template_code_review,
    template_incident_triage,
    template_customer_email,
    template_policy_qa,
    template_roadmap,
    template_runbook,
    template_api_design,
    template_meeting_notes,
    template_docs_howto,
    template_finance_forecast,
    template_security_review,
    template_comparison,
    template_sql_analyst,
]

# Fixed handbook excerpt — shared across requests (prefix-cache realism), substantive not repetitive.
SHARED_HANDBOOK = """Acme Corp assistant handbook (excerpt):
- Answer with structured markdown; lead with the direct answer.
- Never disclose customer PII, unreleased metrics, or credentials.
- When citing policies, name the policy ID if known.
- Prefer actionable steps over generic advice.
- For incidents, separate immediate mitigation from root-cause follow-up.
- For finance or legal topics, state assumptions explicitly.
- Default temperature is 0; be deterministic and concise unless asked otherwise.

Product lines: Orbit Analytics, Nimbus CRM, PulsePay, Vertex Shield, Harbor Logistics, Lumen Docs,
Forge DevOps, Cascade BI. Primary regions: us-central1, us-east1, europe-west4, asia-southeast1.

Escalation: page platform on-call for SEV1; notify customer success within 30 minutes for enterprise tenants.
Data handling: customer content may not be used for model training without written consent and legal review."""


def build_user_prompt(rng: random.Random, request_id: int) -> str:
    template = TEMPLATES[request_id % len(TEMPLATES)]
    # Perturb template choice so consecutive IDs are not always the same family.
    if rng.random() < 0.35:
        template = rng.choice(TEMPLATES)
    return template(rng, request_id)
