"""
type_format_suggester.py
------------------------
The public entry point: suggest(column) -> {"type", "format", "confidence"}.

Pipeline (per the spec's "rules decide, a small model arbitrates"):

    Stage 1 — TYPE (model-backed)
        Load the trained classifier, extract features, predict the semantic
        type and a calibrated confidence via predict_proba.

    Stage 2 — FORMAT (rules + scoring, prior as tie-breaker)
        Only for date / date_time / time / ip_address. Score every allowed
        format by parsing all values; resolve ambiguity with cross-column
        rules (day > 12, "/" => CIDR, ":" => IPv6). If still tied, fall back
        to a documented prior and LOWER the confidence.

    Stage 3 — VALIDATION
        Guarantee type and format are in the allowed sets. Anything off the
        contract is rejected with confidence forced to 0.0 — we never return
        an out-of-set value.

Determinism: no randomness at inference time. Same column -> same output.
"""

from __future__ import annotations

import datetime as _dt
import ipaddress
import os
import re
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

import config
import features as F


# ---------------------------------------------------------------------------
# Model loading (lazy + cached)
# ---------------------------------------------------------------------------
# The model is loaded once and reused. "Inference loads the saved model — no
# retraining at runtime." Loading on every call would be needlessly slow.
_MODEL_CACHE: dict | None = None


def _model_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, config.MODEL_FILENAME)


def _load_model() -> dict:
    """Load and cache the trained model artifact saved by train.py."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        path = _model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found at {path}. Train it first:\n"
                f"    python src/train.py"
            )
        _MODEL_CACHE = joblib.load(path)
    # At this point _MODEL_CACHE is guaranteed to be a dict.
    assert _MODEL_CACHE is not None
    return _MODEL_CACHE


# ---------------------------------------------------------------------------
# Stage 1a — RULE TIER (clear cases, fast + explainable)
# ---------------------------------------------------------------------------
# Per the spec philosophy: "rules handle the clear cases perfectly ... the
# model is a tie-breaker and confidence-calibrator, not the engine." The rule
# tier returns a confident answer ONLY for unambiguous columns; everything
# ambiguous (the 0/1 bool-vs-int problem, overloaded codes, mixed signals)
# falls through to the model.

# Boolean WORDS only — bare "0"/"1" is deliberately excluded, because a 0/1
# column is genuinely ambiguous (bool vs whole_number) and must go to the model.
_BOOL_WORDS = {"true", "false", "yes", "no", "y", "n"}


def _ratio(values: pd.Series, predicate) -> float:
    if len(values) == 0:
        return 0.0
    return float(values.apply(predicate).sum()) / len(values)


def _is_valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_network(value, strict=False) if "/" in value else ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _is_number(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def rule_based_type(clean: pd.Series) -> tuple[str | None, float]:
    """
    Stage 1a: try to type the column with rules alone.

    Returns (type, confidence) for a confident rule hit, else (None, 0.0)
    to signal "ambiguous — let the model decide". Confidence is the match
    ratio itself, so a column with a few malformed values scores slightly
    lower (honest, calibrated).

    Checks run most-specific first so e.g. an IP never gets mistaken for a date.
    """
    thresh = config.RULE_MATCH_MIN_RATIO

    # Email — the "@" structure is unmistakable.
    r = _ratio(clean, lambda v: bool(F.RE_EMAIL.match(v)))
    if r >= thresh:
        return config.TYPE_EMAIL, r

    # IP address — validate with the stdlib for correctness.
    r = _ratio(clean, _is_valid_ip)
    if r >= thresh:
        return config.TYPE_IP_ADDRESS, r

    # Date-time (date AND time together) — before plain date/time.
    r = _ratio(clean, lambda v: bool(F.RE_DATETIME_LIKE.match(v)))
    if r >= thresh:
        return config.TYPE_DATE_TIME, r

    # Time (HH:mm[:ss][ AM/PM]).
    r = _ratio(clean, lambda v: bool(F.RE_TIME.match(v)))
    if r >= thresh:
        return config.TYPE_TIME, r

    # Date (separated date that is not an IP or pure number).
    r = _ratio(clean, lambda v: bool(F.RE_DATE_LIKE.match(v)))
    if r >= thresh:
        return config.TYPE_DATE, r

    # Boolean WORDS (yes/no/true/false) — bare 0/1 intentionally excluded.
    r = _ratio(clean, lambda v: v.lower() in _BOOL_WORDS)
    if r >= thresh:
        return config.TYPE_TRUE_FALSE, r

    # Text fallback: clearly non-numeric, non-pattern values (cities, names,
    # codes). Only fires when almost nothing parses as a number, so numeric
    # columns are left to the model rather than mislabelled text.
    numeric_ratio = _ratio(clean, _is_number)
    if numeric_ratio <= config.TEXT_MAX_NUMERIC_RATIO:
        # Confidence = how clearly non-numeric the column is.
        return config.TYPE_TEXT, round(1.0 - numeric_ratio, 4)

    # Ambiguous (numeric: whole_number vs decimal vs 0/1 bool) — defer to model.
    return None, 0.0



# Each human-facing date/datetime/time label maps to a strptime pattern we can
# actually parse with. Kept right next to the format lists they mirror.
_DATE_STRPTIME = {
    "YYYY-MM-DD": "%Y-%m-%d",
    "DD/MM/YYYY": "%d/%m/%Y",
    "MM/DD/YYYY": "%m/%d/%Y",
    "DD-MM-YYYY": "%d-%m-%Y",
    "MM-DD-YYYY": "%m-%d-%Y",
    "DD.MM.YYYY": "%d.%m.%Y",
    "YYYY/MM/DD": "%Y/%m/%d",
    "DD-MMM-YYYY": "%d-%b-%Y",
    "DD/MM/YY": "%d/%m/%y",
    "MM/DD/YY": "%m/%d/%y",
}
_DATETIME_STRPTIME = {
    "YYYY-MM-DD HH:mm:ss": "%Y-%m-%d %H:%M:%S",
    "YYYY-MM-DDTHH:mm:ss": "%Y-%m-%dT%H:%M:%S",
    "DD/MM/YYYY HH:mm:ss": "%d/%m/%Y %H:%M:%S",
    "MM/DD/YYYY HH:mm:ss": "%m/%d/%Y %H:%M:%S",
    "YYYY-MM-DDTHH:mm:ssZ": "%Y-%m-%dT%H:%M:%SZ",
}
_TIME_STRPTIME = {
    "HH:mm:ss": "%H:%M:%S",
    "HH:mm": "%H:%M",
    "hh:mm:ss A": "%I:%M:%S %p",
    "hh:mm A": "%I:%M %p",
}

# Formats whose first numeric component is the DAY (day-first) vs MONTH-first.
# Used by the day>12 / month>12 disambiguation rules.
_DAY_FIRST_DATE_FORMATS = {"DD/MM/YYYY", "DD-MM-YYYY", "DD.MM.YYYY", "DD/MM/YY"}
_MONTH_FIRST_DATE_FORMATS = {"MM/DD/YYYY", "MM-DD-YYYY", "MM/DD/YY"}

# Documented prior for unresolvable day/month ambiguity. Day-first (DD/MM) is
# the more common civil convention globally, so we default to it — but we LOWER
# confidence when we rely on this rather than on evidence.
_AMBIGUOUS_DATE_PRIOR = "DD/MM/YYYY"


def _strptime_matches(value: str, pattern: str) -> bool:
    try:
        _dt.datetime.strptime(value, pattern)
        return True
    except (ValueError, TypeError):
        return False


def _score_format(values: pd.Series, label: str, strptime_map: dict) -> float:
    """Fraction of values that parse cleanly under this format label."""
    pattern = strptime_map[label]
    if len(values) == 0:
        return 0.0
    matched = values.apply(lambda v: _strptime_matches(v, pattern)).sum()
    return float(matched) / len(values)


def _first_two_components(value: str) -> tuple[int, int] | None:
    """Return the first two integer components of a separated date string."""
    nums = re.findall(r"\d+", value)
    if len(nums) < 2:
        return None
    return int(nums[0]), int(nums[1])


def _detect_date_format(values: pd.Series) -> tuple[str, bool]:
    """
    Decide the best date format.

    Returns (format_label, resolved) where `resolved` is False when we had to
    fall back to the prior (caller lowers confidence in that case).

    Strategy:
        1. Score every candidate by parse success.
        2. Use the day>12 / month>12 rule to eliminate impossible orderings.
        3. Among survivors, take the best score. If day-first and month-first
           remain genuinely tied (e.g. all components <= 12), use the prior
           and report resolved=False.
    """
    scores = {fmt: _score_format(values, fmt, _DATE_STRPTIME)
              for fmt in config.DATE_FORMATS}

    # Keep only formats that parse essentially all values.
    viable = {f: s for f, s in scores.items() if s >= config.FORMAT_MATCH_MIN_RATIO}
    if not viable:
        # Nothing parses cleanly; return best-scoring as a soft guess (unresolved).
        best = max(scores, key=lambda f: scores[f])
        return best, False

    # Apply the day>12 / month>12 elimination using actual evidence.
    saw_first_gt_12 = False    # first component > 12 => first is a DAY
    saw_second_gt_12 = False   # second component > 12 => second is a DAY
    for v in values:
        comp = _first_two_components(str(v))
        if comp is None:
            continue
        a, b = comp
        if a > config.MAX_MONTH:
            saw_first_gt_12 = True
        if b > config.MAX_MONTH:
            saw_second_gt_12 = True

    day_first_viable = {f for f in viable if f in _DAY_FIRST_DATE_FORMATS}
    month_first_viable = {f for f in viable if f in _MONTH_FIRST_DATE_FORMATS}
    unambiguous_viable = {
        f for f in viable
        if f not in _DAY_FIRST_DATE_FORMATS and f not in _MONTH_FIRST_DATE_FORMATS
    }

    # Evidence-based elimination.
    if saw_first_gt_12:
        # First component can't be a month => month-first is impossible.
        month_first_viable = set()
    if saw_second_gt_12:
        # Second component can't be a month => day-first is impossible.
        day_first_viable = set()

    survivors = day_first_viable | month_first_viable | unambiguous_viable

    # An unambiguous format (e.g. YYYY-MM-DD) that parses everything wins outright.
    if unambiguous_viable:
        best_unambig = max(unambiguous_viable, key=lambda f: scores[f])
        # Only prefer it if nothing day/month-first scores strictly higher.
        best_other = max(survivors, key=lambda f: scores[f])
        if scores[best_unambig] >= scores[best_other]:
            return best_unambig, True

    if not survivors:
        return _AMBIGUOUS_DATE_PRIOR, False

    # If both day-first and month-first survived (no >12 evidence), it's a
    # genuine tie -> use the prior and mark unresolved.
    if day_first_viable and month_first_viable:
        return _AMBIGUOUS_DATE_PRIOR, False

    best = max(survivors, key=lambda f: scores[f])
    return best, True


def _detect_ip_format(values: pd.Series) -> tuple[str, bool]:
    """
    Decide the IP format from structural evidence:
        "/" present  => CIDR variant
        ":" present  => IPv6 family
    Validates with the stdlib ipaddress module for correctness.
    """
    has_cidr = values.apply(lambda v: "/" in str(v)).mean() >= config.FORMAT_MATCH_MIN_RATIO
    has_colon = values.apply(lambda v: ":" in str(v)).mean() >= config.FORMAT_MATCH_MIN_RATIO

    if has_colon:
        return ("IPv6 CIDR" if has_cidr else "IPv6"), True
    return ("IPv4 CIDR" if has_cidr else "IPv4"), True


def _detect_simple_format(
    values: pd.Series, candidates: tuple[str, ...], strptime_map: dict
) -> tuple[str, bool]:
    """Pick the best-scoring format among candidates (date_time / time)."""
    scores = {fmt: _score_format(values, fmt, strptime_map) for fmt in candidates}
    best = max(scores, key=lambda f: scores[f])
    resolved = scores[best] >= config.FORMAT_MATCH_MIN_RATIO
    return best, resolved


def detect_format(predicted_type: str, values: pd.Series) -> tuple[str | None, bool]:
    """
    Stage 2 entry point.

    Returns (format_label_or_None, resolved). `resolved=False` signals the
    caller to lower confidence (we leaned on a prior, not evidence).
    Types without formats return (None, True).
    """
    if predicted_type not in config.TYPES_REQUIRING_FORMAT:
        return None, True

    if predicted_type == config.TYPE_DATE:
        return _detect_date_format(values)
    if predicted_type == config.TYPE_IP_ADDRESS:
        return _detect_ip_format(values)
    if predicted_type == config.TYPE_DATE_TIME:
        return _detect_simple_format(values, config.DATE_TIME_FORMATS, _DATETIME_STRPTIME)
    if predicted_type == config.TYPE_TIME:
        return _detect_simple_format(values, config.TIME_FORMATS, _TIME_STRPTIME)

    return None, True


# ---------------------------------------------------------------------------
# Stage 3 — VALIDATION
# ---------------------------------------------------------------------------
def validate(result_type: str, result_format: str | None) -> bool:
    """
    Enforce the output contract:
        - type must be an allowed type
        - if the type requires a format, the format must be in that type's list
        - otherwise the format must be None
    """
    if result_type not in config.ALLOWED_TYPES:
        return False
    if result_type in config.TYPES_REQUIRING_FORMAT:
        return result_format in config.FORMATS_BY_TYPE[result_type]
    return result_format is None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _clean(values: Iterable) -> pd.Series:
    s = pd.Series(list(values), dtype="object").dropna().astype(str).str.strip()
    return s[s != ""]


def suggest(column: pd.Series) -> dict:
    """
    Suggest the semantic type and format for one column.

    Args:
        column: a pandas Series. Its `.name` (if set) is used as a weak signal.

    Returns:
        {"type": str, "format": str | None, "confidence": float}
        with confidence in [0.0, 1.0].
    """
    name = "" if column.name is None else str(column.name)
    values = list(column)
    clean = _clean(values)

    # Empty / all-null columns are not_applicable by definition.
    if len(clean) == 0:
        return {"type": config.TYPE_NOT_APPLICABLE, "format": None, "confidence": 1.0}

    # --- Stage 1: type ---
    # 1a. Rules first (clear cases): fast, explainable, high-confidence.
    predicted_type, confidence = rule_based_type(clean)

    # 1b. Model arbitrates only when the rules can't decide (ambiguous columns).
    if predicted_type is None:
        artifact = _load_model()
        model = artifact["model"]
        label_encoder = artifact["label_encoder"]

        feat_vec = np.asarray(
            [F.extract_feature_vector(name, values)], dtype=float
        )
        proba = model.predict_proba(feat_vec)[0]
        best_idx = int(np.argmax(proba))
        predicted_type = str(label_encoder.inverse_transform([best_idx])[0])
        confidence = float(proba[best_idx])

    # --- Stage 2: format via rules ---
    fmt, resolved = detect_format(predicted_type, clean)
    if not resolved:
        # We leaned on a prior, not evidence: reflect that uncertainty.
        confidence = max(
            config.CONFIDENCE_MIN,
            confidence - config.AMBIGUOUS_FORMAT_CONFIDENCE_PENALTY,
        )

    # --- Stage 3: validate the contract ---
    if not validate(predicted_type, fmt):
        return {
            "type": predicted_type,
            "format": fmt,
            "confidence": config.CONFIDENCE_ON_INVALID,
        }

    # Clamp confidence into [0, 1] defensively.
    confidence = float(min(config.CONFIDENCE_MAX, max(config.CONFIDENCE_MIN, confidence)))
    return {"type": predicted_type, "format": fmt, "confidence": round(confidence, 4)}