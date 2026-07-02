"""Vercel serverless API for the candidate ranking pipeline.

Provides a FastAPI application that wraps the pipeline for Vercel deployment.
Supports uploading candidate samples (JSON/JSONL) and receiving ranked results
as JSON or downloadable CSV.

Endpoints:
  GET  /api/health           — Health check
  POST /api/rank             — Rank uploaded candidates, returns JSON
  POST /api/rank/csv         — Rank uploaded candidates, returns CSV file
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import composite_scorer, reasoning_generator, time_anchor
from rank import build_output_rows


app = FastAPI(
    title="Candidate Ranker API",
    description="Rank candidates against a Senior AI Engineer JD using the same pipeline as rank.py",
    version="1.1.0",
)


def _parse_candidate_file(content: str) -> list[dict]:
    """Parse a JSONL string or JSON array string into candidate dicts."""
    stripped = content.strip()
    if not stripped:
        return []
    if stripped[0] == "[":
        return json.loads(stripped)
    return [
        json.loads(line)
        for line in stripped.splitlines()
        if line.strip()
    ]


def _rank_candidates(candidates: list[dict], top_n: int = 100) -> dict:
    """Run the full ranking pipeline and return results metadata + rows."""
    anchor = time_anchor.compute_time_anchor(candidates)
    scored = composite_scorer.rank_candidates(
        candidates, anchor, top_n=top_n, exclude_honeypots=True
    )
    rows = build_output_rows(scored)

    excluded = sum(
        1
        for c in candidates
        if composite_scorer.compute_candidate_score(c, anchor).trap.is_honeypot
    )

    return {
        "success": True,
        "anchor_date": anchor.date,
        "total_candidates": len(candidates),
        "honeypots_excluded": excluded,
        "ranked_count": len(rows),
        "results": rows,
    }


@app.get("/api/health")
async def health_check():
    """Verify the API is running and pipeline modules are importable."""
    return {
        "status": "ok",
        "pipeline_version": "1.1.0",
    }


@app.post("/api/rank")
async def rank_candidates(
    file: UploadFile = File(...),
    top_n: int = Form(100),
):
    """Upload a candidate sample (JSONL or JSON array) and receive ranked results as JSON.

    Args:
        file: A .jsonl or .json file containing candidate records.
        top_n: Number of top candidates to return (default 100, max 500).

    Returns:
        JSON object with ranked results and metadata.
    """
    content = (await file.read()).decode("utf-8")
    candidates = _parse_candidate_file(content)

    if not candidates:
        return Response(
            content=json.dumps({"success": False, "error": "No valid candidates found in file"}),
            media_type="application/json",
            status_code=400,
        )

    top_n = max(1, min(top_n, 500, len(candidates)))
    result = _rank_candidates(candidates, top_n=top_n)
    return result


@app.post("/api/rank/csv")
async def rank_candidates_csv(
    file: UploadFile = File(...),
    top_n: int = Form(100),
):
    """Upload a candidate sample and receive ranked results as a downloadable CSV.

    Args:
        file: A .jsonl or .json file containing candidate records.
        top_n: Number of top candidates to return (default 100, max 500).

    Returns:
        A CSV file download with columns: candidate_id, rank, score, reasoning.
    """
    content = (await file.read()).decode("utf-8")
    candidates = _parse_candidate_file(content)

    if not candidates:
        return Response(
            content=json.dumps({"success": False, "error": "No valid candidates found in file"}),
            media_type="application/json",
            status_code=400,
        )

    top_n = max(1, min(top_n, 500, len(candidates)))
    result = _rank_candidates(candidates, top_n=top_n)

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["candidate_id", "rank", "score", "reasoning"],
    )
    writer.writeheader()
    writer.writerows(result["results"])

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=ranked_candidates_top_{top_n}.csv",
        },
    )
