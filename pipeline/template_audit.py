"""Hand-graded reference tiers for all 44 career-description templates.

During exploratory data analysis, every ``career_history[].description`` in
the 100k-candidate pool was found to be one of exactly 44 canonical strings.
This discovery collapsed the semantic-matching problem into an exhaustively
auditable one: hand-grade all 44 templates against the JD and validate that
the general scorer in ``evidence_scorer.py`` independently reproduces those
tiers.

Important: this table is NOT used at ranking time. It exists purely as a
validation and calibration reference:
  - ``tests/test_evidence_scorer_vs_audit.py`` verifies the scorer lands on
    the same tier for all 44 templates.
  - ``evaluation_harness.py`` uses it to build a local proxy ground truth for
    sanity-checking the pipeline before submission.

Templates are matched by a 45-character prefix (confirmed unique across all 44
in tests). The full text of each template lives in
``dataset/career_description_templates.json``.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import configuration as config


@dataclass(frozen=True)
class AuditedTemplateEntry:
    """One entry in the hand-graded template audit."""

    template_id: int
    prefix: str
    tier: int
    target_grade: float
    family_label: str
    grading_notes: str


def grade_to_tier(evidence_grade: float) -> int:
    """Map a continuous evidence grade in [0, 1] to an integer tier 0..5.

    Uses the tier band boundaries defined in ``configuration.py``. Bands are
    calibrated so each audited template's target grade falls within its
    intended tier.
    """
    for upper_bound, tier in config.EVIDENCE_TIER_BANDS:
        if evidence_grade < upper_bound:
            return tier
    return 5


AUDITED_TEMPLATES: list[AuditedTemplateEntry] = [
    # ---- Tier 0: non-technical archetypes (keyword-stuffer trap population) ----
    AuditedTemplateEntry(0, "enterprise sales of cloud software solutions", 0, 0.02,
        "sales", "Enterprise SaaS sales; quota carrier. No engineering, ML or IR content."),
    AuditedTemplateEntry(1, "customer support team lead at a saas product", 0, 0.02,
        "customer_support", "Support team lead; explicitly 'lighter on technical depth'. Not engineering."),
    AuditedTemplateEntry(2, "marketing leadership role at a b2b saas company", 0, 0.02,
        "marketing", "Demand-gen / marketing leadership. SEO/content, no ML or IR."),
    AuditedTemplateEntry(3, "business analyst at a consulting firm", 0, 0.03,
        "consulting_ba", "Consulting BA; 'AI-strategy advisory but my own technical depth in AI is limited'."),
    AuditedTemplateEntry(4, "brand design and creative direction at a consu", 0, 0.02,
        "design", "Brand/creative design. No technical relevance."),
    AuditedTemplateEntry(5, "mechanical engineering design role at a hardwa", 0, 0.02,
        "mechanical", "Mechanical/CAD engineering. Non-software; JD wants NLP/IR."),
    AuditedTemplateEntry(6, "senior accounting role at a mid-sized company", 0, 0.02,
        "accounting", "Accounting/finance close. No technical relevance."),
    AuditedTemplateEntry(7, "content writing and seo strategy for a tech-fo", 0, 0.03,
        "content_seo", "Content writer mentioning AI/ML topics and LLM tools — classic keyword bait."),
    AuditedTemplateEntry(8, "operations management role at a logistics comp", 0, 0.02,
        "operations", "Logistics/fulfillment ops management. No technical relevance."),

    # ---- Tier 1: general software engineering (real eng, no ML/IR) -------------
    AuditedTemplateEntry(9, "cloud infrastructure and devops work at an ent", 1, 0.12,
        "devops", "Cloud/DevOps infra; 'haven't done much application development'. Real eng, no ML/IR."),
    AuditedTemplateEntry(10, "android mobile development using java", 1, 0.10,
        "mobile", "Android/mobile dev. Engineering, no ML/IR; explicitly mobile-only."),
    AuditedTemplateEntry(11, "frontend engineering at a media company", 1, 0.09,
        "frontend", "Frontend/React; 'limited backend exposure'. No ML/IR."),
    AuditedTemplateEntry(12, "java backend development at a large enterprise", 1, 0.12,
        "backend", "Java/Spring backend. Solid engineering, no ML/IR."),
    AuditedTemplateEntry(13, "full-stack web application development at a saa", 1, 0.13,
        "fullstack", "Full-stack web dev. General engineering, no ML/IR."),
    AuditedTemplateEntry(14, "test automation and qa engineering for a finte", 1, 0.09,
        "qa", "QA/test automation; 'entirely in QA/test engineering'. No ML/IR."),

    # ---- Tier 2: data engineering / analytics (adjacent, no real ML) ----------
    AuditedTemplateEntry(15, "designed and maintained the analytical data wa", 2, 0.24,
        "data_eng", "Analytics data warehouse / dbt / SQL. Data-adjacent, no modeling."),
    AuditedTemplateEntry(16, "built and maintained data pipelines on apache", 2, 0.27,
        "data_eng", "Airflow/Spark pipelines 'support a few internal ML models' — adjacent, not ML."),
    AuditedTemplateEntry(17, "backend + data hybrid role at a growth-stage s", 2, 0.26,
        "data_eng", "Backend+data warehouse; 'a couple of small predictive features' but mostly data infra."),
    AuditedTemplateEntry(18, "implemented streaming data pipelines on kafka", 2, 0.27,
        "data_eng", "Kafka/Spark streaming; 'some adjacent ML exposure'. Data engineering."),
    AuditedTemplateEntry(19, "mixed data science and analytics-engineering r", 3, 0.40,
        "ds_light", "~30% lightweight ML (clustering/churn in sklearn/XGBoost) + A/B experimentation. Genuine but light ML."),
    AuditedTemplateEntry(20, "backend development with python (fastapi)", 2, 0.22,
        "backend_ml_integ", "Backend that *integrates* a model-serving service 'not the model itself'. Minimal ML."),

    # ---- Tier 3: ML-adjacent modeling (real ML, not retrieval/ranking) ---------
    AuditedTemplateEntry(21, "contributed to ml feature engineering and mode", 3, 0.46,
        "ml_prod_eng", "Production ML engineer (fraud): serving API, feature store, observability. Real ML production."),
    AuditedTemplateEntry(22, "built recommendation-style features at a mid-s", 4, 0.67,
        "recsys", "Production recsys: collaborative filtering + gradient-boosted re-ranking. Relevant but scoped."),
    AuditedTemplateEntry(23, "built computer vision models for our product's", 1, 0.16,
        "cv_primary", "CV-primary (image moderation, ResNet); limited NLP/LLM experience. JD de-prioritizes CV."),
    AuditedTemplateEntry(24, "worked on time-series forecasting models", 3, 0.38,
        "ml_forecasting", "Time-series forecasting (Prophet/LightGBM/LSTM) + RL. Real ML, not IR/ranking."),
    AuditedTemplateEntry(25, "worked on customer-facing predictive modeling", 3, 0.42,
        "ml_predictive", "Predictive modeling (churn/LTV) with sklearn/XGBoost at e-commerce. Applied ML."),
    AuditedTemplateEntry(26, "built nlp pipelines for sentiment analysis", 4, 0.62,
        "nlp_classification", "Production NLP with transformers (DistilBERT, PyTorch/HF). Core JD competency."),

    # ---- Tier 4: strong production ranking / retrieval / MLOps -----------------
    AuditedTemplateEntry(27, "owned the ranking layer for an e-commerce sear", 5, 0.85,
        "ranking_search", "Owned search product ranking layer; hand-tuned -> LTR; relevance labeling + eval. Core JD."),
    AuditedTemplateEntry(28, "trained and shipped multiple ranking models", 5, 0.84,
        "ranking_recsys", "Shipped ranking models for discovery feed; offline-online correlation. Core JD."),
    AuditedTemplateEntry(29, "developed a semantic search feature for an inte", 5, 0.90,
        "semantic_search", "Semantic search with sentence-transformers + FAISS vs BM25, human relevance judgments."),
    AuditedTemplateEntry(30, "implemented a rag-based customer support chatb", 5, 0.85,
        "rag", "RAG chatbot: embeddings + Pinecone + eval framework + measured production impact. Tier 5."),
    AuditedTemplateEntry(31, "built a content recommendation system serving", 5, 0.89,
        "recsys_ranking", "10M-user content recsys: CF + content-based ranking + embeddings + A/B. Core recsys/ranking."),
    AuditedTemplateEntry(32, "built and operated production ml pipelines usin", 4, 0.64,
        "mlops", "Production MLOps (MLflow/Kubeflow/feature store/monitoring). Strong but churn, not ranking."),

    # ---- Tier 5: elite explicit retrieval + ranking at scale ------------------
    AuditedTemplateEntry(33, "built a rag-based ranking pipeline serving 50m", 5, 0.99,
        "elite_recruiter_search", "Hybrid BM25+dense + LLM re-ranker + LTR + NDCG/MRR eval vs A/B. Verbatim the JD."),
    AuditedTemplateEntry(34, "fine-tuned llama-2-7b and mistral-7b variants", 5, 0.93,
        "elite_llm_matching", "LoRA/QLoRA fine-tuning for candidate-JD matching + eval harness + production serving."),
    AuditedTemplateEntry(35, "built and shipped a production recommendation", 5, 0.90,
        "recsys_full", "Marketplace recsys: CF + content + behavioral re-ranking + cold-start + A/B. Full-stack recsys."),
    AuditedTemplateEntry(36, "owned the end-to-end ranking pipeline at a reco", 5, 0.97,
        "elite_hybrid_ranking", "End-to-end ranking: embeddings -> Pinecone -> LTR re-scoring + eval calibration. Elite hybrid."),
    AuditedTemplateEntry(37, "led the migration from keyword-based to embeddi", 5, 0.95,
        "elite_search_migration", "Keyword->embedding search migration 30M-corpus, A/B, index/embedding versioning."),
    AuditedTemplateEntry(38, "owned the design and rollout of a large-scale s", 5, 0.96,
        "elite_semantic_scale", "35M-item semantic search, BM25->hybrid sparse+dense, NDCG@10, index refresh."),

    # ---- Tier 5: plain-language elites (no jargon — the JD's stated trap) ----
    AuditedTemplateEntry(39, "built systems that understand what users are lo", 5, 0.86,
        "plain_matching", "Plain-language: 'connect users to the most relevant matches', heuristic->modeling. Tier 5, zero AI keywords."),
    AuditedTemplateEntry(40, "designed the ranking layer for the company's fl", 5, 0.85,
        "plain_ranking", "Plain-language ranking: 'surface the right thing', owned data pipeline + eval. No jargon."),
    AuditedTemplateEntry(41, "shipped the personalization infrastructure", 5, 0.84,
        "plain_personalization", "Plain-language personalization/relevance + offline/online experimentation + drift detection."),
    AuditedTemplateEntry(42, "owned the search and discovery experience end-t", 5, 0.87,
        "plain_search", "Plain-language search & discovery: 'most relevant results', ranking + evaluation methodology."),
    AuditedTemplateEntry(43, "led the engineering team building infrastructu", 5, 0.85,
        "plain_retrieval", "Plain-language retrieval at scale: 'surface relevant content', billions of docs, index refresh + ranking calibration."),
]


_MATCH_PREFIX_LENGTH: int = 45


def normalize_text_for_matching(text: str) -> str:
    """Lowercase and collapse whitespace for robust prefix matching."""
    return " ".join(text.lower().split())


def match_description_to_audit(description: str) -> AuditedTemplateEntry | None:
    """Return the audit entry whose prefix matches the given description.

    Used only by tests and the proxy-ground-truth builder — never in the
    ranking path. Returns ``None`` for text that is not one of the known
    canonical templates.
    """
    normalized = normalize_text_for_matching(description)
    for entry in AUDITED_TEMPLATES:
        if normalized.startswith(entry.prefix[:_MATCH_PREFIX_LENGTH]):
            return entry
    return None


def build_audit_lookup_table() -> dict[int, AuditedTemplateEntry]:
    """Return a dict mapping template_id -> AuditedTemplateEntry."""
    return {entry.template_id: entry for entry in AUDITED_TEMPLATES}
