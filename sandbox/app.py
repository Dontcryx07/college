"""Streamlit demo for the candidate ranking pipeline.

Deploy on Streamlit Community Cloud:
    streamlit run sandbox/app.py

Runs the exact same pipeline as ``rank.py`` against a small uploaded sample
so reviewers can interact with the ranking logic without the full dataset.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import composite_scorer, reasoning_generator, time_anchor
from rank import build_output_rows


st.set_page_config(page_title="Candidate Ranker", layout="wide")
st.title("Candidate Ranker")
st.caption(
    "Upload a candidate sample (JSONL or JSON array, ≤100 records). "
    "Evidence is read from career descriptions (never the skills list), "
    "honeypots are filtered out, and each row gets fact-grounded reasoning."
)

uploaded_file = st.file_uploader("Candidate sample (.jsonl or .json)", type=["jsonl", "json"])


def _parse_uploaded(file) -> list[dict]:
    raw = file.read().decode("utf-8").strip()
    if not raw:
        return []
    if raw[0] == "[":
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


candidates: list[dict] = []
if uploaded_file is not None:
    candidates = _parse_uploaded(uploaded_file)

if candidates:
    top_n = st.slider("How many to rank", 1, min(100, len(candidates)), min(20, len(candidates)))
    anchor = time_anchor.compute_time_anchor(candidates)
    scored = composite_scorer.rank_candidates(candidates, anchor, top_n=top_n, exclude_honeypots=True)
    rows = build_output_rows(scored)

    st.subheader(f"Top {len(rows)} (anchor date {anchor.date})")
    st.dataframe(rows, use_container_width=True)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    writer.writerows(rows)
    st.download_button("Download ranked CSV", buf.getvalue(), file_name="submission_sample.csv", mime="text/csv")

    excluded = sum(1 for c in candidates if composite_scorer.compute_candidate_score(c, anchor).trap.is_honeypot)
    st.info(f"Honeypots detected and excluded: {excluded}")
else:
    st.write("Upload a candidate file to begin.")
