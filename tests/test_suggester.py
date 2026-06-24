from __future__ import annotations

import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src import config
from type_format_suggester import (  # type: ignore
    detect_format,
    rule_based_type,
    suggest,
    validate,
    _clean,
)


def col(name: str, values: list) -> pd.Series:
    return pd.Series(values, name=name)


def assert_suggestion(
    name: str,
    values: list,
    expected_type: str,
    expected_format: str | None = None,
    min_confidence: float = 0.70,
) -> dict:
    result = suggest(col(name, values))
    assert result["type"] == expected_type, (
        f"[{name}] type: got {result['type']!r}, expected {expected_type!r}"
    )
    assert result["format"] == expected_format, (
        f"[{name}] format: got {result['format']!r}, expected {expected_format!r}"
    )
    assert result["confidence"] >= min_confidence, (
        f"[{name}] confidence {result['confidence']} < {min_confidence}"
    )
    assert config.CONFIDENCE_MIN <= result["confidence"] <= config.CONFIDENCE_MAX, (
        f"[{name}] confidence {result['confidence']} outside [0, 1]"
    )
    return result


class TestTypeMapping:
    """Every type must be detectable and must appear in ALLOWED_TYPES."""

    def test_email(self):
        assert_suggestion(
            "email",
            ["alice@example.com", "bob@ey.com", "carol@gmail.com",
             "david@outlook.com", "eva@startup.io"],
            config.TYPE_EMAIL,
        )

    def test_ip_address_ipv4(self):
        assert_suggestion(
            "login_ip",
            ["10.0.0.1", "192.168.1.20", "172.16.0.5",
             "8.8.8.8", "1.1.1.1", "203.0.113.5"],
            config.TYPE_IP_ADDRESS,
            "IPv4",
        )

    def test_date(self):
        assert_suggestion(
            "txn_date",
            ["16/04/2026", "17/04/2026", "25/12/2025",
             "01/03/2024", "30/06/2023"],
            config.TYPE_DATE,
            "DD/MM/YYYY",
        )

    def test_date_time(self):
        assert_suggestion(
            "created_at",
            ["2026-04-16 10:30:00", "2026-04-17 11:00:00",
             "2025-12-25 00:00:00", "2024-01-01 08:15:30"],
            config.TYPE_DATE_TIME,
            "YYYY-MM-DD HH:mm:ss",
        )

    def test_time(self):
        assert_suggestion(
            "start_time",
            ["09:30:00", "14:15:00", "23:59:59",
             "00:00:00", "12:00:00", "07:45:30"],
            config.TYPE_TIME,
            "HH:mm:ss",
        )

    def test_true_false_words(self):
        assert_suggestion(
            "is_active",
            ["yes", "no", "yes", "yes", "no", "yes", "no", "yes"],
            config.TYPE_TRUE_FALSE,
        )

    def test_true_false_tf(self):
        assert_suggestion(
            "verified",
            ["true", "false", "true", "true", "false",
             "true", "false", "true", "false"],
            config.TYPE_TRUE_FALSE,
        )

    def test_whole_number(self):
        assert_suggestion(
            "count",
            ["12", "45", "7", "103", "88", "5", "200", "34", "91", "16"],
            config.TYPE_WHOLE_NUMBER,
        )

    def test_decimal_number(self):
        assert_suggestion(
            "amount",
            ["85000.50", "72000.00", "61000.25", "95000.75",
             "78500.10", "67000.33", "91000.00", "55000.99"],
            config.TYPE_DECIMAL_NUMBER,
        )

    def test_text(self):
        assert_suggestion(
            "city",
            ["London", "Paris", "Berlin", "Tokyo",
             "Madrid", "Rome", "Oslo", "Lima"],
            config.TYPE_TEXT,
        )

    def test_not_applicable_empty(self):
        result = suggest(col("blank", ["", "", "", ""]))
        assert result["type"] == config.TYPE_NOT_APPLICABLE
        assert result["format"] is None
        assert result["confidence"] == 1.0

    def test_not_applicable_nulls(self):
        result = suggest(col("nulls", [None, None, None, None]))
        assert result["type"] == config.TYPE_NOT_APPLICABLE

    def test_all_types_in_allowed(self):
        test_columns = [
            col("e", ["a@b.com"] * 5),
            col("d", ["2026-01-01"] * 5),
            col("n", ["123"] * 5),
            col("t", ["hello world"] * 5),
            col("b", ["yes", "no"] * 4),
        ]
        for c in test_columns:
            r = suggest(c)
            assert r["type"] in config.ALLOWED_TYPES, (
                f"suggest() returned unknown type: {r['type']!r}"
            )


class TestDateFormats:
    """Every allowed date format must be detected correctly."""

    @pytest.mark.parametrize("fmt,values", [
        ("YYYY-MM-DD",   ["2026-04-16", "2025-12-25", "2024-03-30",
                          "2023-07-04", "2022-11-11"]),
        ("DD/MM/YYYY",   ["16/04/2026", "25/12/2025", "30/03/2024",
                          "04/07/2023", "11/11/2022"]),
        ("MM/DD/YYYY",   ["04/16/2026", "12/25/2025", "03/30/2024",
                          "07/04/2023", "11/30/2022"]),
        ("DD-MM-YYYY",   ["16-04-2026", "25-12-2025", "30-03-2024",
                          "04-07-2023", "11-11-2022"]),
        ("MM-DD-YYYY",   ["04-16-2026", "12-25-2025", "03-30-2024",
                          "07-14-2023", "11-30-2022"]),
        ("DD.MM.YYYY",   ["16.04.2026", "25.12.2025", "30.03.2024",
                          "04.07.2023", "11.11.2022"]),
        ("YYYY/MM/DD",   ["2026/04/16", "2025/12/25", "2024/03/30",
                          "2023/07/04", "2022/11/11"]),
        ("DD-MMM-YYYY",  ["16-Apr-2026", "25-Dec-2025", "30-Mar-2024",
                          "04-Jul-2023", "11-Nov-2022"]),
        ("DD/MM/YY",     ["16/04/26", "25/12/25", "30/03/24",
                          "14/07/23", "11/11/22"]),
        ("MM/DD/YY",     ["04/16/26", "12/25/25", "03/30/24",
                          "07/14/23", "11/30/22"]),
    ])
    def test_date_format(self, fmt, values):
        result = suggest(col("date_col", values))
        assert result["type"] == config.TYPE_DATE, (
            f"Expected date, got {result['type']!r} for format {fmt}"
        )
        assert result["format"] == fmt, (
            f"Expected format {fmt!r}, got {result['format']!r}"
        )
        assert result["confidence"] >= 0.70


class TestDateTimeFormats:
    """Every allowed datetime format must be detected."""

    @pytest.mark.parametrize("fmt,values", [
        ("YYYY-MM-DD HH:mm:ss",
         ["2026-04-16 10:30:00", "2025-12-25 00:00:00", "2024-03-30 08:15:30"]),
        ("YYYY-MM-DDTHH:mm:ss",
         ["2026-04-16T10:30:00", "2025-12-25T00:00:00", "2024-03-30T08:15:30"]),
        ("DD/MM/YYYY HH:mm:ss",
         ["16/04/2026 10:30:00", "25/12/2025 00:00:00", "30/03/2024 08:15:30"]),
        ("MM/DD/YYYY HH:mm:ss",
         ["04/16/2026 10:30:00", "12/25/2025 00:00:00", "03/30/2024 08:15:30"]),
        ("YYYY-MM-DDTHH:mm:ssZ",
         ["2026-04-16T10:30:00Z", "2025-12-25T00:00:00Z", "2024-03-30T08:15:30Z"]),
    ])
    def test_datetime_format(self, fmt, values):
        result = suggest(col("ts_col", values))
        assert result["type"] == config.TYPE_DATE_TIME, (
            f"Expected date_time, got {result['type']!r} for format {fmt}"
        )
        assert result["format"] == fmt, (
            f"Expected format {fmt!r}, got {result['format']!r}"
        )


class TestTimeFormats:
    """Every allowed time format must be detected."""

    @pytest.mark.parametrize("fmt,values", [
        ("HH:mm:ss",    ["09:30:00", "14:15:00", "23:59:59", "00:00:00"]),
        ("HH:mm",       ["09:30", "14:15", "23:59", "00:00", "12:00"]),
        ("hh:mm:ss A",  ["09:30:00 AM", "02:15:00 PM", "11:59:59 PM", "12:00:00 AM"]),
        ("hh:mm A",     ["09:30 AM", "02:15 PM", "11:59 PM", "12:00 AM"]),
    ])
    def test_time_format(self, fmt, values):
        result = suggest(col("time_col", values))
        assert result["type"] == config.TYPE_TIME, (
            f"Expected time, got {result['type']!r} for format {fmt}"
        )
        assert result["format"] == fmt, (
            f"Expected format {fmt!r}, got {result['format']!r}"
        )


class TestIPFormats:
    """Every allowed IP format must be detected."""

    @pytest.mark.parametrize("fmt,values", [
        ("IPv4",      ["10.0.0.1", "192.168.1.20", "172.16.0.5", "8.8.8.8"]),
        ("IPv4 CIDR", ["10.0.0.0/24", "192.168.1.0/16", "172.16.0.0/12"]),
        ("IPv6",      ["2001:db8::1", "fe80::1", "::1", "2001:db8::2"]),
        ("IPv6 CIDR", ["2001:db8::/32", "fe80::/64", "2001:db8:1::/48"]),
    ])
    def test_ip_format(self, fmt, values):
        result = suggest(col("ip_col", values))
        assert result["type"] == config.TYPE_IP_ADDRESS, (
            f"Expected ip_address, got {result['type']!r} for format {fmt}"
        )
        assert result["format"] == fmt, (
            f"Expected format {fmt!r}, got {result['format']!r}"
        )

class TestDateDisambiguation:
    def test_day_gt_12_eliminates_month_first(self):
        result = suggest(col("d", [
            "16/04/2026",
            "25/12/2025",
            "30/03/2024",
            "14/07/2023",
        ]))
        assert result["type"] == config.TYPE_DATE
        assert result["format"] == "DD/MM/YYYY", (
            f"Day > 12 in first slot must resolve to DD/MM/YYYY, "
            f"got {result['format']!r}"
        )
        assert result["confidence"] >= 0.90, (
            "A resolvable date should have high confidence"
        )

    def test_month_gt_12_second_position_eliminates_day_first(self):
        result = suggest(col("d", [
            "04/16/2026",
            "12/25/2025",
            "03/30/2024",
            "07/14/2023",
        ]))
        assert result["type"] == config.TYPE_DATE
        assert result["format"] == "MM/DD/YYYY", (
            f"Value > 12 in second slot must resolve to MM/DD/YYYY, "
            f"got {result['format']!r}"
        )
        assert result["confidence"] >= 0.90

    def test_unresolvable_uses_prior_and_lowers_confidence(self):
        """
        When ALL values have both components <= 12, DD/MM vs MM/DD is
        genuinely unresolvable. We must:
          - fall back to the documented prior (DD/MM/YYYY)
          - lower confidence by AMBIGUOUS_FORMAT_CONFIDENCE_PENALTY
        """
        result = suggest(col("d", [
            "01/02/2026",
            "03/04/2025",
            "05/06/2024",
            "07/08/2023",
            "02/11/2022",
        ]))
        assert result["type"] == config.TYPE_DATE
        assert result["format"] == "DD/MM/YYYY", (
            "Unresolvable date must fall back to the DD/MM/YYYY prior"
        )
        assert result["confidence"] < 0.90, (
            f"Unresolvable date confidence {result['confidence']} should be "
            f"lower than 0.90 to reflect the uncertainty"
        )

    def test_iso_date_unambiguous(self):
        result = suggest(col("d", [
            "2026-04-16", "2025-12-25", "2024-03-30", "2023-07-04",
        ]))
        assert result["type"] == config.TYPE_DATE
        assert result["format"] == "YYYY-MM-DD"
        assert result["confidence"] >= 0.90

    def test_ipv4_cidr_slash_present(self):
        result = suggest(col("subnet", [
            "10.0.0.0/24", "192.168.1.0/16", "172.16.0.0/12",
        ]))
        assert result["format"] == "IPv4 CIDR"

    def test_ipv6_colon_present(self):
        result = suggest(col("ip6", [
            "2001:db8::1", "fe80::1", "::1", "2001:db8::2",
        ]))
        assert result["format"] in ("IPv6", "IPv6 CIDR")

class TestModelTieBreaker:
    def test_binary_column_routes_to_model(self):
        result = suggest(col("flag", ["0", "1", "1", "0", "1", "0", "0", "1", "1", "0"]))
        assert result["type"] in (config.TYPE_TRUE_FALSE, config.TYPE_WHOLE_NUMBER), (
            f"0/1 column must resolve to true_false or whole_number, "
            f"got {result['type']!r}"
        )
        assert result["confidence"] < 1.0, (
            "0/1 column is genuinely ambiguous — confidence must be < 1.0"
        )

    def test_named_flag_column_leans_true_false(self):
        result = suggest(col("is_active", ["0", "1", "1", "0", "1", "0", "0", "1"]))
        assert result["type"] == config.TYPE_TRUE_FALSE
        assert result["confidence"] < 1.0

    def test_model_confidence_in_range(self):
        for values in [
            ["0", "1", "0", "1", "1", "0"],
            ["42", "17", "8", "200", "3", "91"],
            ["3.14", "2.71", "1.41", "1.73"],
        ]:
            result = suggest(col("x", values))
            assert config.CONFIDENCE_MIN <= result["confidence"] <= config.CONFIDENCE_MAX



class TestValidator:
    def test_valid_type_no_format(self):
        assert validate(config.TYPE_TEXT, None) is True

    def test_valid_type_with_correct_format(self):
        assert validate(config.TYPE_DATE, "DD/MM/YYYY") is True
        assert validate(config.TYPE_IP_ADDRESS, "IPv4") is True
        assert validate(config.TYPE_TIME, "HH:mm") is True

    def test_invalid_type_rejected(self):
        assert validate("banana", None) is False
        assert validate("integer", None) is False      # old name, not in list
        assert validate("", None) is False

    def test_invalid_format_rejected(self):
        assert validate(config.TYPE_DATE, "YYYY/DD/MM") is False   # not in list
        assert validate(config.TYPE_DATE, "banana") is False
        assert validate(config.TYPE_IP_ADDRESS, "IPv5") is False

    def test_format_none_for_non_temporal(self):
        assert validate(config.TYPE_TEXT, None) is True
        assert validate(config.TYPE_TEXT, "DD/MM/YYYY") is False   # text has no format
        assert validate(config.TYPE_EMAIL, None) is True
        assert validate(config.TYPE_WHOLE_NUMBER, "YYYY-MM-DD") is False

    def test_temporal_type_requires_format(self):
        assert validate(config.TYPE_DATE, None) is False
        assert validate(config.TYPE_TIME, None) is False
        assert validate(config.TYPE_IP_ADDRESS, None) is False

class TestConfidenceContract:
    @pytest.mark.parametrize("values,name", [
        (["alice@example.com"] * 8,  "email"),
        (["10.0.0.1"] * 8,           "ip"),
        (["2026-01-01"] * 8,         "date"),
        (["yes", "no"] * 4,          "flag"),
        (["London"] * 8,             "city"),
        (["0", "1"] * 4,             "binary"),
        (["42"] * 8,                 "num"),
        (["", ""] * 4,               "empty"),
        (["3.14", "2.71"] * 4,       "float"),
        (["09:30:00"] * 8,           "time"),
    ])
    def test_confidence_in_range(self, values, name):
        result = suggest(col(name, values))
        assert config.CONFIDENCE_MIN <= result["confidence"] <= config.CONFIDENCE_MAX, (
            f"[{name}] confidence {result['confidence']} outside [0, 1]"
        )


class TestEdgeCases:
    def test_single_value_column(self):
        """A column with one value must not crash."""
        result = suggest(col("x", ["hello@world.com"]))
        assert "type" in result
        assert "format" in result
        assert "confidence" in result

    def test_all_nulls(self):
        result = suggest(col("x", [None, None, None]))
        assert result["type"] == config.TYPE_NOT_APPLICABLE
        assert result["confidence"] == 1.0

    def test_mixed_nulls_and_values(self):
        result = suggest(col("e", [
            "alice@example.com", None, "bob@ey.com", None, "carol@gmail.com",
        ]))
        assert result["type"] == config.TYPE_EMAIL

    def test_result_has_required_keys(self):
        result = suggest(col("x", ["hello", "world"]))
        assert set(result.keys()) == {"type", "format", "confidence"}

    def test_column_without_name(self):
        s = pd.Series(["London", "Paris", "Berlin"])
        result = suggest(s)
        assert result["type"] == config.TYPE_TEXT

    def test_dirty_email_column(self):
        result = suggest(col("email", [
            "alice@example.com",
            "bob@ey.com",
            "carol@gmail.com",
            "notanemail",
            "david@outlook.com",
            "eva@startup.io",
            "frank@corp.co",
            "grace@webmail.net",
            "henry@company.org",
            "iris@naver.com",
            "jake@acme.co",
        ]))
        assert result["type"] == config.TYPE_EMAIL

    def test_rule_tier_returns_none_for_ambiguous(self):
        clean = _clean(["42", "17", "8", "200", "3", "91", "55", "12"])
        detected_type, confidence = rule_based_type(clean)
        assert detected_type is None, (
            "Pure integer column should fall through to model, "
            f"not be resolved by rules as {detected_type!r}"
        )
        assert confidence == 0.0