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


_MODEL_CACHE: dict | None = None


def _model_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, config.MODEL_FILENAME)


def _load_model() -> dict:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        path = _model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found at {path}. Train it first:\n"
                f"    python src/train.py"
            )
        _MODEL_CACHE = joblib.load(path)
    assert _MODEL_CACHE is not None
    return _MODEL_CACHE

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
    thresh = config.RULE_MATCH_MIN_RATIO

    r = _ratio(clean, lambda v: bool(F.RE_EMAIL.match(v)))
    if r >= thresh:
        return config.TYPE_EMAIL, r

    r = _ratio(clean, _is_valid_ip)
    if r >= thresh:
        return config.TYPE_IP_ADDRESS, r

    r = _ratio(clean, lambda v: bool(F.RE_DATETIME_LIKE.match(v)))
    if r >= thresh:
        return config.TYPE_DATE_TIME, r

    r = _ratio(clean, lambda v: bool(F.RE_TIME.match(v)))
    if r >= thresh:
        return config.TYPE_TIME, r

    r = _ratio(clean, lambda v: bool(F.RE_DATE_LIKE.match(v)))
    if r >= thresh:
        return config.TYPE_DATE, r

    r = _ratio(clean, lambda v: v.lower() in _BOOL_WORDS)
    if r >= thresh:
        return config.TYPE_TRUE_FALSE, r

    numeric_ratio = _ratio(clean, _is_number)
    if numeric_ratio <= config.TEXT_MAX_NUMERIC_RATIO:
        return config.TYPE_TEXT, round(1.0 - numeric_ratio, 4)

    return None, 0.0

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


_DAY_FIRST_DATE_FORMATS = {"DD/MM/YYYY", "DD-MM-YYYY", "DD.MM.YYYY", "DD/MM/YY"}
_MONTH_FIRST_DATE_FORMATS = {"MM/DD/YYYY", "MM-DD-YYYY", "MM/DD/YY"}

_AMBIGUOUS_DATE_PRIOR = "DD/MM/YYYY"


def _strptime_matches(value: str, pattern: str) -> bool:
    try:
        _dt.datetime.strptime(value, pattern)
        return True
    except (ValueError, TypeError):
        return False


def _score_format(values: pd.Series, label: str, strptime_map: dict) -> float:
    pattern = strptime_map[label]
    if len(values) == 0:
        return 0.0
    matched = values.apply(lambda v: _strptime_matches(v, pattern)).sum()
    return float(matched) / len(values)


def _first_two_components(value: str) -> tuple[int, int] | None:
    nums = re.findall(r"\d+", value)
    if len(nums) < 2:
        return None
    return int(nums[0]), int(nums[1])


def _detect_date_format(values: pd.Series) -> tuple[str, bool]:
    scores = {fmt: _score_format(values, fmt, _DATE_STRPTIME)
              for fmt in config.DATE_FORMATS}

    viable = {f: s for f, s in scores.items() if s >= config.FORMAT_MATCH_MIN_RATIO}
    if not viable:
        if not scores:
            return "", False
        best = max(scores, key=scores.get) # type: ignore
        return best, False


    saw_first_gt_12 = False
    saw_second_gt_12 = False
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

    if saw_first_gt_12:
        month_first_viable = set()
    if saw_second_gt_12:
        day_first_viable = set()

    survivors = day_first_viable | month_first_viable | unambiguous_viable

    if unambiguous_viable:
        best_unambig = max(unambiguous_viable, key=lambda f: scores[f])
        best_other = max(survivors, key=lambda f: scores[f])
        if scores[best_unambig] >= scores[best_other]:
            return best_unambig, True

    if not survivors:
        return _AMBIGUOUS_DATE_PRIOR, False

    if day_first_viable and month_first_viable:
        return _AMBIGUOUS_DATE_PRIOR, False

    best = max(survivors, key=lambda f: scores[f])
    return best, True


def _detect_ip_format(values: pd.Series) -> tuple[str, bool]:
    has_cidr = values.apply(lambda v: "/" in str(v)).mean() >= config.FORMAT_MATCH_MIN_RATIO
    has_colon = values.apply(lambda v: ":" in str(v)).mean() >= config.FORMAT_MATCH_MIN_RATIO

    if has_colon:
        return ("IPv6 CIDR" if has_cidr else "IPv6"), True
    return ("IPv4 CIDR" if has_cidr else "IPv4"), True


def _detect_simple_format(
    values: pd.Series, candidates: tuple[str, ...], strptime_map: dict
) -> tuple[str, bool]:
    scores = {fmt: _score_format(values, fmt, strptime_map) for fmt in candidates}
    best = max(scores, key=scores.get) # type: ignore
    resolved = scores[best] >= config.FORMAT_MATCH_MIN_RATIO
    return best, resolved


def detect_format(predicted_type: str, values: pd.Series) -> tuple[str | None, bool]:
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


def validate(result_type: str, result_format: str | None) -> bool:
    if result_type not in config.ALLOWED_TYPES:
        return False
    if result_type in config.TYPES_REQUIRING_FORMAT:
        return result_format in config.FORMATS_BY_TYPE[result_type]
    return result_format is None


def _clean(values: Iterable) -> pd.Series:
    s = pd.Series(list(values), dtype="object").dropna().astype(str).str.strip()
    return s[s != ""]


def suggest(column: pd.Series) -> dict:
    name = "" if column.name is None else str(column.name)
    values = list(column)
    clean = _clean(values)

    if len(clean) == 0:
        return {"type": config.TYPE_NOT_APPLICABLE, "format": None, "confidence": 1.0}

    predicted_type, confidence = rule_based_type(clean)

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

    fmt, resolved = detect_format(predicted_type, clean)
    if not resolved:
        confidence = max(
            config.CONFIDENCE_MIN,
            confidence - config.AMBIGUOUS_FORMAT_CONFIDENCE_PENALTY,
        )

    if not validate(predicted_type, fmt):
        return {
            "type": predicted_type,
            "format": fmt,
            "confidence": config.CONFIDENCE_ON_INVALID,
        }

    confidence = float(min(config.CONFIDENCE_MAX, max(config.CONFIDENCE_MIN, confidence)))
    return {"type": predicted_type, "format": fmt, "confidence": round(confidence, 4)}