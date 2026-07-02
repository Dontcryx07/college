# Redrob Candidate Ranker — Intelligent Candidate Discovery & Ranking Challenge

A pure-Python, zero-dependency candidate ranking system for the Redrob AI Hackathon. Ranks 100K candidates against the Senior AI Engineer — Founding Team job description in ~20 seconds on CPU.

## Quick start

```bash
# 1. Ensure Python 3.10+ is installed
# 2. Install dependencies (only pytest for local validation)
pip install -r requirements.txt

# 3. Unpack the candidate pool (gzipped file in dataset/)
#    (if candidates.jsonl.gz is present)
python -c "import gzip; gzip.open('dataset/candidates.jsonl.gz','rb'); open('dataset/candidates.jsonl','wb').write(__import__('gzip').open('dataset/candidates.jsonl.gz','rb').read())"

# 4. Produce the submission CSV
python rank.py --candidates ./dataset/candidates.jsonl --out ./submission.csv
```

The single command to reproduce the submission:

```bash
python rank.py --candidates ./dataset/candidates.jsonl --out ./submission.csv
```

## Output

The output CSV (`submission.csv`) contains exactly 100 rows with columns:
`candidate_id`, `rank`, `score`, `reasoning` — matching the submission spec.

## Validation

```bash
python dataset/validate_submission.py submission.csv
```

## Architecture

| Module | Purpose |
|--------|---------|
| `rank.py` | Entrypoint — CLI, orchestration, CSV export |
| `pipeline/evidence_scorer.py` | Ontology phrase matching against career descriptions (never the skills list). Collapses ~44 unique templates via memoization. |
| `pipeline/template_audit.py` | Hand-graded reference tiers for all 44 career-description templates (used only for validation, not ranking). |
| `pipeline/job_fit_scorer.py` | JD logistics fit — experience band, location tier, notice period, product vs. consulting, tenure stability, ML depth. |
| `pipeline/behavior_analyzer.py` | Behavioral availability multiplier — recency, recruiter response rate, open-to-work, interview completion. |
| `pipeline/trap_detector.py` | Honeypot exclusion (YoE inflation, phantom expert skills) and keyword-stuffer soft penalty. |
| `pipeline/composite_scorer.py` | Tier-lexicographic score combination: evidence_tier + within_tier_score. |
| `pipeline/reasoning_generator.py` | Fact-grounded reasoning text with per-candidate rotation for variety. |
| `pipeline/time_anchor.py` | Deterministic "today" derived from the data (max last_active_date). |
| `pipeline/evaluation_harness.py` | Local proxy metrics (NDCG, MAP, P@K) against a template-audit ground truth. |
| `pipeline/configuration.py` | All tuning constants, weights, thresholds in one place. |

## Scoring formula

```
within_score    = evidence_grade × (1 + 0.30 × JD_fit) × behavior_mult × trap_penalty
composite_score = evidence_tier + squash(within_score / 1.55)
```

Evidence tier (0–5) **dominates** ordering — no amount of location/notice/behavior bonus can push a lower-tier candidate above a higher-tier one.

## Key design decisions

- **Evidence from career descriptions only** — the skills list is the keyword-stuffer trap surface the JD warns about.
- **Plain-language Tier-5 detection** — ontology catches candidates who built ranking/retrieval systems without using AI jargon.
- **Honeypot exclusion** — candidates with YoE inflation or phantom expert skills are dropped before ranking.
- **Zero dependencies** — the ranking pipeline uses only Python stdlib (`json`, `gzip`, `csv`, `re`, `math`, `dataclasses`), satisfying the 5-minute CPU-only constraint.

## Sandbox

A Streamlit sandbox is available at `https://college-hp6yepr2hgh7iytw75bkjb.streamlit.app/`:

```bash
pip install streamlit
streamlit run sandbox/app.py
```

Upload a JSONL/JSON sample (≤100 candidates) to interactively explore rankings.

## Compute constraints

- **Runtime**: ~20s for 100K candidates (well under 5-minute limit)
- **Memory**: ~1.5 GB peak
- **Compute**: CPU only (no GPU)
- **Network**: None (no API calls)
- **Python**: 3.10+
