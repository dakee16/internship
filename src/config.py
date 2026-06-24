from __future__ import annotations

from typing import Final


RANDOM_SEED: Final[int] = 42


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

FORMATS_BY_TYPE: Final[dict[str, tuple[str, ...]]] = {
    TYPE_DATE: DATE_FORMATS,
    TYPE_DATE_TIME: DATE_TIME_FORMATS,
    TYPE_TIME: TIME_FORMATS,
    TYPE_IP_ADDRESS: IP_FORMATS,
}

TYPES_REQUIRING_FORMAT: Final[tuple[str, ...]] = tuple(FORMATS_BY_TYPE.keys())


CONFIDENCE_MIN: Final[float] = 0.0
CONFIDENCE_MAX: Final[float] = 1.0

CONFIDENCE_ON_INVALID: Final[float] = 0.0


AMBIGUOUS_FORMAT_CONFIDENCE_PENALTY: Final[float] = 0.30

MAX_MONTH: Final[int] = 12
MAX_DAY: Final[int] = 31

FORMAT_MATCH_MIN_RATIO: Final[float] = 0.90

RULE_MATCH_MIN_RATIO: Final[float] = 0.90

TEXT_MAX_NUMERIC_RATIO: Final[float] = 0.10

MAX_SAMPLE_SIZE: Final[int] = 500

MODEL_FILENAME: Final[str] = "model.joblib"



def _validate_config() -> None:
    for type_name in FORMATS_BY_TYPE:
        assert type_name in ALLOWED_TYPES, (
            f"FORMATS_BY_TYPE references unknown type: {type_name!r}"
        )

    assert CONFIDENCE_MIN < CONFIDENCE_MAX, "Confidence bounds are inverted."
    assert CONFIDENCE_MIN <= CONFIDENCE_ON_INVALID <= CONFIDENCE_MAX

    assert 0.0 <= FORMAT_MATCH_MIN_RATIO <= 1.0
    assert 0.0 <= AMBIGUOUS_FORMAT_CONFIDENCE_PENALTY <= 1.0


_validate_config()