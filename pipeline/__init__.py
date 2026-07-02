"""Candidate ranking pipeline for the Redrob AI Hackathon.

This package implements a pure-Python, zero-dependency candidate ranking
system that ranks candidates against the Senior AI Engineer — Founding Team
job description. The entire pipeline runs in ~20-25 seconds on CPU.

Architecture:
  - ``configuration.py`` — all tuning constants, thresholds, and weights
  - ``data_access.py`` — candidate IO, safe accessors, date utilities
  - ``time_anchor.py`` — deterministic "today" derived from the data
  - ``evidence_scorer.py`` — ontology matching → evidence grade from text
  - ``template_audit.py`` — 44 hand-graded template reference tiers
  - ``trap_detector.py`` — honeypot and keyword-stuffer detection
  - ``job_fit_scorer.py`` — JD logistics fit sub-scores
  - ``behavior_analyzer.py`` — behavioral availability multiplier
  - ``composite_scorer.py`` — final score combination + ranking
  - ``reasoning_generator.py`` — fact-grounded reasoning text
  - ``evaluation_harness.py`` — local proxy evaluation metrics
"""

from __future__ import annotations

__version__ = "1.1.0"
