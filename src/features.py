from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

import config


RE_EMAIL = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
RE_INTEGER = re.compile(r"^[+\-]?\d+$")
RE_DECIMAL = re.compile(r"^[+\-]?(\d+\.\d*|\.\d+|\d+\.\d+)$")
RE_BOOLEAN = re.compile(r"^(true|false|yes|no|y|n|0|1)$", re.IGNORECASE)
RE_IPV4 = re.compile(r"^(\d{1,3}\.){3}\d{1,3}(/\d{1,2})?$")
RE_IPV6 = re.compile(r"^[0-9A-Fa-f:]+:[0-9A-Fa-f:]+(/\d{1,3})?$")
RE_TIME = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?(\s?[APap][Mm])?$")
RE_DATE_LIKE = re.compile(
    r"^(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}"
    r"|\d{1,2}[/\-.][A-Za-z]{3}[/\-.]\d{2,4})$"
)
RE_DATETIME_LIKE = re.compile(
    r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}[ T]\d{1,2}:\d{2}(:\d{2})?Z?$"
)

NAME_SIGNALS = {
    "name_has_date": ("date", "dob", "day"),
    "name_has_time": ("time", "clock", "hour"),
    "name_has_timestamp": ("timestamp", "_at", "datetime"),
    "name_has_id": ("id", "code", "key"),
    "name_has_ip": ("ip", "host", "addr"),
    "name_has_email": ("email", "mail", "contact"),
    "name_has_amount": ("amount", "price", "rate", "total", "cost", "ratio"),
    "name_has_flag": ("is_", "flag", "active", "enabled", "verified"),
}


FEATURE_NAMES: tuple[str, ...] = (
    # --- regex match-ratios ---
    "ratio_email",
    "ratio_integer",
    "ratio_decimal",
    "ratio_boolean",
    "ratio_ipv4",
    "ratio_ipv6",
    "ratio_time",
    "ratio_date_like",
    "ratio_datetime_like",
    # --- statistical ---
    "avg_length",
    "std_length",
    "unique_ratio",
    "null_ratio",
    "numeric_ratio",
    "avg_token_count",
    "avg_digit_ratio",
    "ratio_has_slash",
    "ratio_has_dash",
    "ratio_has_dot",
    "ratio_has_colon",
    "ratio_has_at",
    "ratio_has_space",
    # --- column-name signals ---
    "name_has_date",
    "name_has_time",
    "name_has_timestamp",
    "name_has_id",
    "name_has_ip",
    "name_has_email",
    "name_has_amount",
    "name_has_flag",
)



def _clean_values(values: Iterable) -> pd.Series:
    s = pd.Series(list(values), dtype="object")
    s = s.dropna().astype(str).str.strip()
    return s[s != ""]


def _match_ratio(values: pd.Series, pattern: re.Pattern) -> float:
    if len(values) == 0:
        return 0.0
    matched = values.apply(lambda v: bool(pattern.match(v))).sum()
    return float(matched) / len(values)


def _char_presence_ratio(values: pd.Series, ch: str) -> float:
    if len(values) == 0:
        return 0.0
    return float(values.apply(lambda v: ch in v).sum()) / len(values)


def _digit_ratio(value: str) -> float:
    if len(value) == 0:
        return 0.0
    digits = sum(c.isdigit() for c in value)
    return digits / len(value)


def _is_floaty(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False

def extract_features(name: str, values: Iterable) -> dict[str, float]:
    raw = pd.Series(list(values), dtype="object")
    total = len(raw)

    if total > config.MAX_SAMPLE_SIZE:
        raw = raw.sample(config.MAX_SAMPLE_SIZE, random_state=config.RANDOM_SEED)

    clean = _clean_values(raw)
    n_clean = len(clean)

    null_ratio = 0.0 if len(raw) == 0 else 1.0 - (n_clean / len(raw))

    feats: dict[str, float] = {}

    # --- regex match-ratios ---
    feats["ratio_email"] = _match_ratio(clean, RE_EMAIL)
    feats["ratio_integer"] = _match_ratio(clean, RE_INTEGER)
    feats["ratio_decimal"] = _match_ratio(clean, RE_DECIMAL)
    feats["ratio_boolean"] = _match_ratio(clean, RE_BOOLEAN)
    feats["ratio_ipv4"] = _match_ratio(clean, RE_IPV4)
    feats["ratio_ipv6"] = _match_ratio(clean, RE_IPV6)
    feats["ratio_time"] = _match_ratio(clean, RE_TIME)
    feats["ratio_date_like"] = _match_ratio(clean, RE_DATE_LIKE)
    feats["ratio_datetime_like"] = _match_ratio(clean, RE_DATETIME_LIKE)

    # --- statistical features ---
    if n_clean > 0:
        lengths = clean.str.len().to_numpy(dtype=float)
        feats["avg_length"] = float(np.mean(lengths))
        feats["std_length"] = float(np.std(lengths))
        feats["unique_ratio"] = float(clean.nunique()) / n_clean
        feats["numeric_ratio"] = float(clean.apply(_is_floaty).sum()) / n_clean
        feats["avg_token_count"] = float(
            np.mean([len(v.split()) for v in clean])
        )
        feats["avg_digit_ratio"] = float(np.mean([_digit_ratio(v) for v in clean]))
    else:
        feats["avg_length"] = 0.0
        feats["std_length"] = 0.0
        feats["unique_ratio"] = 0.0
        feats["numeric_ratio"] = 0.0
        feats["avg_token_count"] = 0.0
        feats["avg_digit_ratio"] = 0.0

    feats["null_ratio"] = float(null_ratio)
    feats["ratio_has_slash"] = _char_presence_ratio(clean, "/")
    feats["ratio_has_dash"] = _char_presence_ratio(clean, "-")
    feats["ratio_has_dot"] = _char_presence_ratio(clean, ".")
    feats["ratio_has_colon"] = _char_presence_ratio(clean, ":")
    feats["ratio_has_at"] = _char_presence_ratio(clean, "@")
    feats["ratio_has_space"] = _char_presence_ratio(clean, " ")

    # --- column-name signals ---
    name_lower = (name or "").lower()
    for feat_name, tokens in NAME_SIGNALS.items():
        feats[feat_name] = 1.0 if any(tok in name_lower for tok in tokens) else 0.0

    return feats


def features_to_vector(feats: dict[str, float]) -> list[float]:
    assert set(feats.keys()) == set(FEATURE_NAMES), (
        "Feature keys do not match FEATURE_NAMES. "
        f"Missing: {set(FEATURE_NAMES) - set(feats.keys())}; "
        f"Unexpected: {set(feats.keys()) - set(FEATURE_NAMES)}"
    )
    return [float(feats[name]) for name in FEATURE_NAMES]


def extract_feature_vector(name: str, values: Iterable) -> list[float]:
    return features_to_vector(extract_features(name, values))