"""
config.py
---------
Central configuration for the Column Type & Format Suggester.

Single source of truth for:
    - the allowed semantic TYPES
    - the allowed FORMATS for each temporal / IP type
    - all thresholds (no magic numbers anywhere else in the codebase)
    - the global RANDOM_SEED for deterministic behaviour

Design principle:
    Every other module imports from here. Nothing in features.py,
    type_format_suggester.py, the data generator, or the tests should
    hardcode a type name, a format string, or a numeric threshold.
    Change it once here, and the whole system (and its tests) follow.
"""

from __future__ import annotations

from typing import Final


# ---------------------------------------------------------------------------
# 1.  DETERMINISM
# ---------------------------------------------------------------------------
# A single seed used by the data generator, the train/test split, and the
# model itself. Fixed so results are reproducible across runs and machines.
RANDOM_SEED: Final[int] = 42


# ---------------------------------------------------------------------------
# 2.  SEMANTIC TYPES
# ---------------------------------------------------------------------------
# The 10 allowed semantic types. The model may ONLY ever output one of these,
# and the validator enforces it. `not_applicable` is the catch-all for columns
# that are empty / all-null / unusable.
TYPE_TEXT: Final[str] = "text"
TYPE_WHOLE_NUMBER: Final[str] = "whole_number"
TYPE_DECIMAL_NUMBER: Final[str] = "decimal_number"
TYPE_TRUE_FALSE: Final[str] = "true_false"
TYPE_DATE: Final[str] = "date"
TYPE_DATE_TIME: Final[str] = "date_time"
TYPE_TIME: Final[str] = "time"
TYPE_EMAIL: Final[str] = "email"
TYPE_IP_ADDRESS: Final[str] = "ip_address"
TYPE_NOT_APPLICABLE: Final[str] = "not_applicable"

ALLOWED_TYPES: Final[tuple[str, ...]] = (
    TYPE_TEXT,
    TYPE_WHOLE_NUMBER,
    TYPE_DECIMAL_NUMBER,
    TYPE_TRUE_FALSE,
    TYPE_DATE,
    TYPE_DATE_TIME,
    TYPE_TIME,
    TYPE_EMAIL,
    TYPE_IP_ADDRESS,
    TYPE_NOT_APPLICABLE,
)


# ---------------------------------------------------------------------------
# 3.  FORMATS PER TYPE
# ---------------------------------------------------------------------------
# Formats are ONLY meaningful for the four types below. For every other type,
# the suggested format must be None.
#
# Notation reference (this is the human-facing label we return, NOT a strptime
# string — features.py / the format detector maps these to actual parsers):
#   YYYY = 4-digit year   YY = 2-digit year
#   MM   = 2-digit month  MMM = 3-letter month (Jan)
#   DD   = 2-digit day
#   HH   = 24-hour hour   hh = 12-hour hour    A = AM/PM
#   mm   = minutes        ss = seconds         Z = trailing UTC marker

DATE_FORMATS: Final[tuple[str, ...]] = (
    "YYYY-MM-DD",
    "DD/MM/YYYY",
    "MM/DD/YYYY",
    "DD-MM-YYYY",
    "MM-DD-YYYY",
    "DD.MM.YYYY",
    "YYYY/MM/DD",
    "DD-MMM-YYYY",
    "DD/MM/YY",
    "MM/DD/YY",
)

DATE_TIME_FORMATS: Final[tuple[str, ...]] = (
    "YYYY-MM-DD HH:mm:ss",
    "YYYY-MM-DDTHH:mm:ss",
    "DD/MM/YYYY HH:mm:ss",
    "MM/DD/YYYY HH:mm:ss",
    "YYYY-MM-DDTHH:mm:ssZ",
)

TIME_FORMATS: Final[tuple[str, ...]] = (
    "HH:mm:ss",
    "HH:mm",
    "hh:mm:ss A",
    "hh:mm A",
)

IP_FORMATS: Final[tuple[str, ...]] = (
    "IPv4",
    "IPv4 CIDR",
    "IPv6",
    "IPv6 CIDR",
)

# Lookup: type -> its allowed formats. Types not in this dict take format=None.
FORMATS_BY_TYPE: Final[dict[str, tuple[str, ...]]] = {
    TYPE_DATE: DATE_FORMATS,
    TYPE_DATE_TIME: DATE_TIME_FORMATS,
    TYPE_TIME: TIME_FORMATS,
    TYPE_IP_ADDRESS: IP_FORMATS,
}

# Types that REQUIRE a non-null format. Every other type must have format=None.
TYPES_REQUIRING_FORMAT: Final[tuple[str, ...]] = tuple(FORMATS_BY_TYPE.keys())


# ---------------------------------------------------------------------------
# 4.  THRESHOLDS  (no magic numbers elsewhere)
# ---------------------------------------------------------------------------
# Confidence bounds — the contract guarantees confidence is in [0.0, 1.0].
CONFIDENCE_MIN: Final[float] = 0.0
CONFIDENCE_MAX: Final[float] = 1.0

# When the validator rejects an out-of-set output, confidence is forced here.
CONFIDENCE_ON_INVALID: Final[float] = 0.0

# Penalty applied to confidence when the format detector cannot fully
# disambiguate and has to rely on the model's prior to break a tie.
# (e.g. "01/02/2026" — no number > 12, so DD/MM vs MM/DD is unresolved.)
AMBIGUOUS_FORMAT_CONFIDENCE_PENALTY: Final[float] = 0.30

# Calendar boundary used by the date disambiguation rules:
#   first component > MAX_MONTH  => not a month  => eliminate month-first formats
#   second component > MAX_MONTH => not a month  => eliminate day-first formats
MAX_MONTH: Final[int] = 12
MAX_DAY: Final[int] = 31

# A format candidate is considered a match for the column only if at least
# this fraction of the column's values parse cleanly under it.
FORMAT_MATCH_MIN_RATIO: Final[float] = 0.90

# ---- Rules-first hybrid (Stage 1) ----
# The rule tier fires only when a single unambiguous pattern matches at least
# this fraction of values. Kept high so rules handle ONLY clear cases and the
# model still arbitrates the genuinely ambiguous ones (e.g. 0/1 bool-vs-int).
RULE_MATCH_MIN_RATIO: Final[float] = 0.90

# The text fallback fires only when almost nothing parses as a number, so
# numeric-looking columns are left to the model instead of being called text.
TEXT_MAX_NUMERIC_RATIO: Final[float] = 0.10

# Sampling cap: for very long columns we analyse at most this many values
# (keeps inference fast; deterministic because we seed the sample).
MAX_SAMPLE_SIZE: Final[int] = 500


# ---------------------------------------------------------------------------
# 5.  MODEL / ARTIFACT PATHS
# ---------------------------------------------------------------------------
MODEL_FILENAME: Final[str] = "model.joblib"


# ---------------------------------------------------------------------------
# 6.  SELF-VALIDATION
# ---------------------------------------------------------------------------
def _validate_config() -> None:
    """
    Internal sanity checks run at import time.

    Catches configuration mistakes early (e.g. a format list referencing a
    type that isn't in ALLOWED_TYPES) instead of letting them surface as
    confusing failures deep in the model or validator.
    """
    # Every type that has formats must be a recognised type.
    for type_name in FORMATS_BY_TYPE:
        assert type_name in ALLOWED_TYPES, (
            f"FORMATS_BY_TYPE references unknown type: {type_name!r}"
        )

    # Confidence bounds must be sane.
    assert CONFIDENCE_MIN < CONFIDENCE_MAX, "Confidence bounds are inverted."
    assert CONFIDENCE_MIN <= CONFIDENCE_ON_INVALID <= CONFIDENCE_MAX

    # Ratios must be valid probabilities.
    assert 0.0 <= FORMAT_MATCH_MIN_RATIO <= 1.0
    assert 0.0 <= AMBIGUOUS_FORMAT_CONFIDENCE_PENALTY <= 1.0


_validate_config()