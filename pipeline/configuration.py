"""Central configuration constants for the candidate ranking pipeline.

All tuning parameters, ontology weights, thresholds, and bounds are defined in
this single module so they can be inspected, swept, or overridden without
hunting across six files. Sub-modules import from here rather than defining
their own literals.
"""

from __future__ import annotations

# ============================================================================
# Evidence Scorer — concept ontology weights & thresholds
# ============================================================================
# Positive signal families: name -> (weight, [phrases]).
# Core IR/ranking/recsys families carry the most weight because they are the
# JD's actual target work. Weights are calibrated against the 44-template audit
# (see pipeline/template_audit.py).
POSITIVE_SIGNAL_FAMILIES: dict[str, tuple[float, list[str]]] = {
    "retrieval_search": (0.55, [
        "semantic search", "search product", "search engine", "search and discovery",
        "search relevance", "retrieval", "vector search", "nearest-neighbor",
        "nearest neighbor", "faiss", "pinecone", "milvus", "weaviate", "qdrant",
        "elasticsearch", "opensearch", "bm25", "information retrieval",
        "surface relevant", "surface the right thing", "most relevant results",
        "most relevant matches", "query understanding", "query expansion",
        "keyword-based to embedding", "embedding-based search", "hybrid retrieval",
        "dense retrieval", "sparse and dense", "index refresh",
        "connect them to the most relevant",
    ]),
    "ranking": (0.55, [
        "ranking layer", "ranking model", "ranking models", "ranking pipeline",
        "ranking algorithm", "ranking algorithms", "ranking calibration", "re-rank",
        "re-ranker", "re-ranking", "reranker", "learning-to-rank", "learning to rank",
        "scoring function", "relevance labeling", "discovery feed", "ltr",
    ]),
    "recsys_matching": (0.50, [
        "recommendation system", "recommendation-style", "recommender",
        "recommendations-heavy", "collaborative filtering", "matrix factorization",
        "content-based ranking", "content recommendation", "matching layer",
        "matching system", "personalization", "personalized", "cold-start",
        "cold start", "candidate-jd matching",
    ]),
    "embeddings": (0.28, [
        "embedding", "embeddings", "sentence-transformer", "sentence transformers",
        "bge", "mpnet", "all-minilm", "minilm", "dense vector",
    ]),
    "nlp_llm": (0.20, [
        "nlp", "natural language", "llm", "rag", "retrieval-augmented",
        "transformer", "transformers", "fine-tune", "fine-tuned", "fine tuning",
        "lora", "qlora", "distilbert", "bert", "hugging face", "huggingface",
        "llama", "mistral", "gpt-4", "openai embeddings", "sentiment analysis",
        "document classification", "language model",
    ]),
    "eval_framework": (0.20, [
        "ndcg", "mrr", "recall@", "offline evaluation", "offline metrics",
        "offline-online", "offline and online", "a/b test", "a/b testing",
        "evaluation framework", "evaluation methodology", "relevance judgments",
        "held-out eval", "held-out", "eval harness", "offline experimentation",
        "experimentation framework", "simulated a/b", "online engagement",
        "explicit modeling and evaluation",
    ]),
    "mlops": (0.28, [
        "mlflow", "kubeflow", "feature store", "model serving", "model-serving",
        "drift detection", "embedding drift", "data drift", "model monitoring",
        "bentoml", "index versioning", "embedding versioning", "retraining",
        "inference service", "production ml pipelines",
    ]),
    "ml_general": (0.26, [
        "machine learning", "xgboost", "lightgbm", "scikit-learn", "sklearn",
        "gradient-boosted", "gradient boosted", "predictive model",
        "predictive modeling", "churn prediction", "churn model", "classification",
        "clustering", "forecasting", "feature engineering", "prophet", "lstm",
        "reinforcement learning", "pytorch",
    ]),
    "data_engineering": (0.16, [
        "airflow", "spark", "pyspark", "dbt", "snowflake", "data warehouse",
        "data pipelines", "data pipeline", "data quality", "data infrastructure",
        "looker", "dimensional modeling",
    ]),
    "general_engineering": (0.12, [
        "backend", "microservices", "spring boot", "react", "typescript",
        "full-stack", "fullstack", "kubernetes", "docker", "terraform", "devops",
        "rest api", "android", "kotlin", "frontend", "selenium", "ci/cd",
        "fastapi", "node.js", "postgres",
    ]),
    "scale_indicators": (0.06, [
        "10m+", "50m+", "30m+", "35m+", "millions of", "billions of",
        "queries per month", "500k", "200k",
    ]),
}

NONTECH_KEYWORD_MARKERS: list[str] = [
    "enterprise sales", "sales cycle", "arr quota", "quota", "support agents",
    "customer support team", "demand-generation", "demand generation",
    "content marketing", "performance marketing", "account-based marketing",
    "brand identity", "brand design", "packaging design", "adobe suite",
    "creative direction", "mechanical engineering", "solidworks", "creo",
    "month-end close", "statutory compliance", "fixed-asset", "staff accountants",
    "fulfillment operations", "picking, packing", "warehouses", "consulting firm",
    "business diagnostics", "slide-craft", "content writing", "seo strategy",
    "editorial calendar", "freelance writer",
]

CV_PRIMARY_KEYWORD_MARKERS: list[str] = [
    "computer vision", "image moderation", "resnet", "object detection",
]

HONESTY_DISCLAIMER_MARKERS: list[str] = [
    "adjacent ml exposure", "some adjacent ml", "wouldn't call myself an ml specialist",
    "technical depth in ai is limited", "not the model itself",
    "lighter on technical depth", "professional experience there is limited",
    "my own modeling work was secondary", "haven't done much application development",
    "modeling work was secondary", "limited backend exposure",
    "deployment was handled by", "handled by the platform team", "lighter weight than",
]

# Squashing constant for evidence grade: grade = 1 - exp(-K * raw_score).
EVIDENCE_SQUASH_K: float = 2.2

# Floor grade assigned to real-but-unrelated software engineering work.
GENERAL_ENGINEERING_FLOOR: float = 0.13

# Multiplier applied to grade when honesty disclaimer is detected.
HONESTY_DISCLAIMER_MULTIPLIER: float = 0.72

# Crushing multiplier for non-technical (keyword-bait) roles.
NONTECH_KEYWORD_MULTIPLIER: float = 0.06

# Cap for computer-vision-primary work lacking retrieval/ranking overlap.
CV_PRIMARY_GRADE_CAP: float = 0.16

# Family names treated as floor-only (not additive to raw score).
FLOOR_ONLY_FAMILIES: tuple[str, ...] = ("general_engineering",)

# Families considered core for CV-cap logic.
CORE_EVIDENCE_FAMILIES: tuple[str, ...] = ("retrieval_search", "ranking", "recsys_matching")

# Thresholds for strong role detection.
STRONG_ROLE_GRADE_THRESHOLD: float = 0.58
ML_ROLE_GRADE_THRESHOLD: float = 0.36

# ============================================================================
# Trap Detector — honeypot and keyword-stuffer thresholds
# ============================================================================
# Years-of-experience inflation threshold.
YOE_INFLATION_GAP_THRESHOLD: float = 3.0

# Minimum number of phantom expert skills (expert proficiency, 0 duration) to flag.
PHANTOM_EXPERT_MINIMUM_COUNT: int = 3

# Minimum AI/ML skill count to consider keyword-stuffer signal.
AI_SKILL_STUFFER_THRESHOLD: int = 5

# Evidence grade below which keyword-stuffer flag activates.
KEYWORD_STUFFER_EVIDENCE_CEILING: float = 0.20

# ============================================================================
# Job Description Fit — sub-score weights and thresholds
# ============================================================================
# Weight applied to JD-fit in the final multiplier: (1 + ALPHA * jd_fit_score).
JD_FIT_ALPHA: float = 0.30

# Location tier keywords (lowercased).
PUNE_NOIDA_CITIES: tuple[str, ...] = ("pune", "noida")
INDIA_TIER_ONE_CITIES: tuple[str, ...] = (
    "hyderabad", "mumbai", "bangalore", "bengaluru", "delhi", "gurgaon",
    "gurugram", "chennai", "new delhi",
)

# Sub-score weights for JD-fit blending (sum used for normalization).
JD_FIT_SUBSCORE_WEIGHTS: dict[str, float] = {
    "experience": 0.34,
    "location": 0.30,
    "notice": 0.15,
    "product": 0.11,
    "skill_trust": 0.10,
    "tenure": 0.10,
    "ml_depth": 0.12,
}

# Product company markers in candidate descriptions.
PRODUCT_COMPANY_MARKERS: tuple[str, ...] = (
    "product company", "consumer product", "consumer-product", "marketplace",
    "b2b saas", "saas product", "saas company", "growth-stage startup",
    "mid-stage startup", "e-commerce", "consumer-app", "consumer app",
    "flagship product",
)
CONSULTING_COMPANY_MARKER: str = "consulting firm"

# Experience fit band thresholds.
EXPERIENCE_PEAK_LOW: float = 6.0
EXPERIENCE_PEAK_HIGH: float = 8.0

# Tenure (stability) thresholds in months.
TENURE_EXCELLENT_MONTHS: float = 36.0
TENURE_GOOD_MONTHS: float = 24.0
TENURE_NEUTRAL_MONTHS: float = 18.0
TENURE_MINIMUM_MONTHS: float = 12.0

# ML depth thresholds in years.
ML_DEPTH_EXCELLENT_YEARS: float = 4.0
ML_DEPTH_GOOD_YEARS: float = 3.0
ML_DEPTH_FAIR_YEARS: float = 2.0
ML_DEPTH_MINIMUM_YEARS: float = 1.0

# ============================================================================
# Behavioral Analyzer — multiplier bounds and component weights
# ============================================================================
BEHAVIOR_MINIMUM_MULTIPLIER: float = 0.60
BEHAVIOR_MAXIMUM_MULTIPLIER: float = 1.10

BEHAVIOR_COMPONENT_WEIGHTS: dict[str, float] = {
    "recency": 0.40,
    "response": 0.30,
    "open_to_work": 0.20,
    "interview": 0.10,
}

# ============================================================================
# Composite Scorer — final score combination
# ============================================================================
STUFFER_PENALTY_MULTIPLIER: float = 0.5
WITHIN_SCORE_DIVISOR: float = 1.55

# ============================================================================
# Tier band boundaries for evidence grade -> integer tier mapping
# ============================================================================
EVIDENCE_TIER_BANDS: list[tuple[float, int]] = [
    (0.05, 0),
    (0.20, 1),
    (0.36, 2),
    (0.58, 3),
    (0.78, 4),
    (1.01, 5),
]
