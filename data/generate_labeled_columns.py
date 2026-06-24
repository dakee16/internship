from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from src import config  # noqa: E402


COLUMNS_PER_CLASS: int = 200
MIN_COL_LEN: int = 8
MAX_COL_LEN: int = 40
NULL_INJECT_PROB: float = 0.15
NULL_CELL_PROB: float = 0.10
NULL_TOKEN: str = ""

YEAR_MIN: int = 1990
YEAR_MAX: int = 2030

NUMBER_MESSY_PROB: float = 0.35
EMAIL_CORRUPT_PROB: float = 0.08
TEXT_CODEY_PROB: float = 0.40
AMBIGUOUS_FRACTION: float = 0.50

CURRENCY_SYMBOLS = ["$", "€", "£", "₹"]
UNITS = ["kg", "lb", "cm", "ms", "MB", "%"]

CODE_PREFIXES = ["SKU", "REF", "ORD", "TKT", "v", "build", "ext"]

FIRST_NAMES = ["alice", "bob", "carol", "david", "eva", "frank", "grace",
               "henry", "iris", "jake", "kira", "liam", "mary", "noah"]
LAST_NAMES = ["johnson", "smith", "white", "lee", "martin", "brown", "kim",
              "davis", "lopez", "wilson", "patel", "garcia", "khan", "singh"]
CITIES = ["london", "paris", "berlin", "tokyo", "mumbai", "sydney", "toronto",
          "madrid", "rome", "dubai", "oslo", "lima", "cairo", "seoul"]
DEPARTMENTS = ["engineering", "finance", "hr", "marketing", "sales", "legal"]
FREE_TEXT = ["great performance this quarter", "needs follow up next week",
             "approved by manager", "pending review", "see attached notes",
             "escalated to support", "no action required", "on hold until q3"]
EMAIL_DOMAINS = ["example.com", "ey.com", "gmail.com", "outlook.com",
                 "company.org", "webmail.net", "startup.io", "corp.co"]
MONTHS_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

NAME_HINTS = {
    config.TYPE_TEXT: ["city", "name", "notes", "comment", "description", "label"],
    config.TYPE_WHOLE_NUMBER: ["count", "quantity", "age", "items", "score", "id"],
    config.TYPE_DECIMAL_NUMBER: ["amount", "price", "rate", "weight", "total", "ratio"],
    config.TYPE_TRUE_FALSE: ["is_active", "flag", "enabled", "verified", "paid"],
    config.TYPE_DATE: ["date", "txn_date", "dob", "start_date", "created_on"],
    config.TYPE_DATE_TIME: ["timestamp", "created_at", "logged_at", "event_time"],
    config.TYPE_TIME: ["time", "start_time", "clock", "alarm", "shift_time"],
    config.TYPE_EMAIL: ["email", "contact", "user_email", "login", "mail"],
    config.TYPE_IP_ADDRESS: ["ip", "login_ip", "host", "server_ip", "client_ip"],
    config.TYPE_NOT_APPLICABLE: ["col", "field", "x", "unnamed", "blank"],
}
NEUTRAL_NAMES = ["col_a", "column1", "field", "value", "data", "x", "y"]


def _rand_year(rng: random.Random) -> int:
    return rng.randint(YEAR_MIN, YEAR_MAX)


def _rand_day(rng: random.Random, max_day: int = 28) -> int:
    return rng.randint(1, max_day)


def _rand_month(rng: random.Random) -> int:
    return rng.randint(1, config.MAX_MONTH)


def _fmt_date(fmt: str, y: int, m: int, d: int) -> str:
    yy = y % 100
    mmm = MONTHS_ABBR[m - 1]
    return {
        "YYYY-MM-DD": f"{y:04d}-{m:02d}-{d:02d}",
        "DD/MM/YYYY": f"{d:02d}/{m:02d}/{y:04d}",
        "MM/DD/YYYY": f"{m:02d}/{d:02d}/{y:04d}",
        "DD-MM-YYYY": f"{d:02d}-{m:02d}-{y:04d}",
        "MM-DD-YYYY": f"{m:02d}-{d:02d}-{y:04d}",
        "DD.MM.YYYY": f"{d:02d}.{m:02d}.{y:04d}",
        "YYYY/MM/DD": f"{y:04d}/{m:02d}/{d:02d}",
        "DD-MMM-YYYY": f"{d:02d}-{mmm}-{y:04d}",
        "DD/MM/YY": f"{d:02d}/{m:02d}/{yy:02d}",
        "MM/DD/YY": f"{m:02d}/{d:02d}/{yy:02d}",
    }[fmt]


def _fmt_time(fmt: str, h24: int, mi: int, se: int) -> str:
    ampm = "AM" if h24 < 12 else "PM"
    h12 = h24 % 12
    if h12 == 0:
        h12 = 12
    return {
        "HH:mm:ss": f"{h24:02d}:{mi:02d}:{se:02d}",
        "HH:mm": f"{h24:02d}:{mi:02d}",
        "hh:mm:ss A": f"{h12:02d}:{mi:02d}:{se:02d} {ampm}",
        "hh:mm A": f"{h12:02d}:{mi:02d} {ampm}",
    }[fmt]


def _gen_text(rng: random.Random) -> str:
    if rng.random() < TEXT_CODEY_PROB:
        return _gen_codey_text(rng)
    pool = rng.choice([CITIES, DEPARTMENTS, FREE_TEXT,
                       [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"]])
    return rng.choice(pool)


def _gen_whole_number(rng: random.Random) -> str:
    n = rng.randint(0, 100_000)
    if rng.random() >= NUMBER_MESSY_PROB:
        return str(n)
    style = rng.choice(["comma", "currency", "unit", "plain"])
    if style == "comma":
        return f"{n:,}"
    if style == "currency":
        return f"{rng.choice(CURRENCY_SYMBOLS)}{n:,}"
    if style == "unit":
        return f"{n} {rng.choice(UNITS)}"
    return str(n)


def _gen_decimal_number(rng: random.Random) -> str:
    val = rng.uniform(0, 100_000)
    plain = f"{val:.{rng.randint(1, 4)}f}"
    if rng.random() >= NUMBER_MESSY_PROB:
        return plain
    style = rng.choice(["comma", "currency", "percent", "unit"])
    if style == "comma":
        return f"{val:,.2f}"
    if style == "currency":
        return f"{rng.choice(CURRENCY_SYMBOLS)}{val:,.2f}"
    if style == "percent":
        return f"{rng.uniform(0, 100):.1f}%"
    return f"{plain} {rng.choice(UNITS)}"


def _gen_codey_text(rng: random.Random) -> str:
    style = rng.choice(["sku", "version", "phone", "alnum"])
    if style == "sku":
        return f"{rng.choice(CODE_PREFIXES)}-{rng.randint(1000, 9999)}"
    if style == "version":
        return f"v{rng.randint(0,9)}.{rng.randint(0,20)}.{rng.randint(0,99)}"
    if style == "phone":
        return f"+{rng.randint(1,99)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}"
    return f"{rng.choice(string.ascii_uppercase)}{rng.randint(100,999)}{rng.choice(string.ascii_uppercase)}"


def _gen_true_false(rng: random.Random) -> str:
    pair = rng.choice([("true", "false"), ("yes", "no"),
                       ("y", "n"), ("0", "1"), ("True", "False")])
    return rng.choice(pair)


def _gen_email(rng: random.Random) -> str:
    user = rng.choice(FIRST_NAMES)
    sep = rng.choice([".", "_", ""])
    extra = rng.choice(["", str(rng.randint(1, 99)), rng.choice(LAST_NAMES)])
    email = f"{user}{sep}{extra}@{rng.choice(EMAIL_DOMAINS)}"
    if rng.random() < EMAIL_CORRUPT_PROB:
        corruption = rng.choice(["no_at", "no_domain", "double_dot"])
        if corruption == "no_at":
            return email.replace("@", "")
        if corruption == "no_domain":
            return email.split("@")[0]
        return email.replace(".", "..", 1)
    return email


def _gen_ipv4(rng: random.Random) -> str:
    return ".".join(str(rng.randint(0, 255)) for _ in range(4))


def _gen_ipv4_cidr(rng: random.Random) -> str:
    return f"{_gen_ipv4(rng)}/{rng.randint(8, 32)}"


def _gen_ipv6(rng: random.Random) -> str:
    return ":".join(f"{rng.randint(0, 65535):x}" for _ in range(8))


def _gen_ipv6_cidr(rng: random.Random) -> str:
    return f"{_gen_ipv6(rng)}/{rng.randint(16, 128)}"


def _maybe_inject_nulls(values: list[str], rng: random.Random) -> list[str]:
    if rng.random() >= NULL_INJECT_PROB:
        return values
    return [NULL_TOKEN if rng.random() < NULL_CELL_PROB else v for v in values]


def _pick_name(type_name: str, rng: random.Random) -> str:
    if rng.random() < 0.25:
        return rng.choice(NEUTRAL_NAMES)
    return rng.choice(NAME_HINTS[type_name])


def _build_column(
    type_name: str,
    fmt: str | None,
    value_fn: Callable[[random.Random], str],
    rng: random.Random,
    ambiguity: str = "none",
) -> dict:
    n = rng.randint(MIN_COL_LEN, MAX_COL_LEN)
    values = [value_fn(rng) for _ in range(n)]
    values = _maybe_inject_nulls(values, rng)
    return {
        "name": _pick_name(type_name, rng),
        "values": values,
        "type": type_name,
        "format": fmt,
        "ambiguity": ambiguity,
    }


def _build_date_column(fmt: str, rng: random.Random, ambiguity: str) -> dict:
    n = rng.randint(MIN_COL_LEN, MAX_COL_LEN)
    values: list[str] = []

    for _ in range(n):
        y = _rand_year(rng)
        if ambiguity == "unresolvable":
            d = rng.randint(1, config.MAX_MONTH)
            m = rng.randint(1, config.MAX_MONTH)
        else:
            d = _rand_day(rng, max_day=28)
            m = _rand_month(rng)
        values.append(_fmt_date(fmt, y, m, d))

    if ambiguity == "resolvable":
        y = _rand_year(rng)
        big_day = rng.randint(config.MAX_MONTH + 1, 28)
        m = _rand_month(rng)
        idx = rng.randrange(len(values))
        values[idx] = _fmt_date(fmt, y, m, big_day)

    values = _maybe_inject_nulls(values, rng)
    return {
        "name": _pick_name(config.TYPE_DATE, rng),
        "values": values,
        "type": config.TYPE_DATE,
        "format": fmt,
        "ambiguity": ambiguity,
    }


def _build_datetime_column(fmt: str, rng: random.Random) -> dict:
    n = rng.randint(MIN_COL_LEN, MAX_COL_LEN)
    values = []
    for _ in range(n):
        y, m, d = _rand_year(rng), _rand_month(rng), _rand_day(rng)
        h, mi, se = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
        date_part = {
            "YYYY-MM-DD HH:mm:ss": f"{y:04d}-{m:02d}-{d:02d} {h:02d}:{mi:02d}:{se:02d}",
            "YYYY-MM-DDTHH:mm:ss": f"{y:04d}-{m:02d}-{d:02d}T{h:02d}:{mi:02d}:{se:02d}",
            "DD/MM/YYYY HH:mm:ss": f"{d:02d}/{m:02d}/{y:04d} {h:02d}:{mi:02d}:{se:02d}",
            "MM/DD/YYYY HH:mm:ss": f"{m:02d}/{d:02d}/{y:04d} {h:02d}:{mi:02d}:{se:02d}",
            "YYYY-MM-DDTHH:mm:ssZ": f"{y:04d}-{m:02d}-{d:02d}T{h:02d}:{mi:02d}:{se:02d}Z",
        }[fmt]
        values.append(date_part)
    values = _maybe_inject_nulls(values, rng)
    return {
        "name": _pick_name(config.TYPE_DATE_TIME, rng),
        "values": values,
        "type": config.TYPE_DATE_TIME,
        "format": fmt,
        "ambiguity": "none",
    }


def _build_time_column(fmt: str, rng: random.Random) -> dict:
    n = rng.randint(MIN_COL_LEN, MAX_COL_LEN)
    values = []
    for _ in range(n):
        h, mi, se = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
        values.append(_fmt_time(fmt, h, mi, se))
    values = _maybe_inject_nulls(values, rng)
    return {
        "name": _pick_name(config.TYPE_TIME, rng),
        "values": values,
        "type": config.TYPE_TIME,
        "format": fmt,
        "ambiguity": "none",
    }


def _build_not_applicable_column(rng: random.Random) -> dict:
    n = rng.randint(MIN_COL_LEN, MAX_COL_LEN)
    blank = rng.choice([NULL_TOKEN, " ", "  "])
    return {
        "name": _pick_name(config.TYPE_NOT_APPLICABLE, rng),
        "values": [blank for _ in range(n)],
        "type": config.TYPE_NOT_APPLICABLE,
        "format": None,
        "ambiguity": "none",
    }


def generate_dataset(
    columns_per_class: int = COLUMNS_PER_CLASS,
    seed: int = config.RANDOM_SEED,
) -> list[dict]:
    rng = random.Random(seed)
    dataset: list[dict] = []

    simple_generators = {
        config.TYPE_TEXT: _gen_text,
        config.TYPE_WHOLE_NUMBER: _gen_whole_number,
        config.TYPE_DECIMAL_NUMBER: _gen_decimal_number,
        config.TYPE_TRUE_FALSE: _gen_true_false,
        config.TYPE_EMAIL: _gen_email,
    }
    for type_name, fn in simple_generators.items():
        for _ in range(columns_per_class):
            dataset.append(_build_column(type_name, None, fn, rng))

    ip_generators = {
        "IPv4": _gen_ipv4,
        "IPv4 CIDR": _gen_ipv4_cidr,
        "IPv6": _gen_ipv6,
        "IPv6 CIDR": _gen_ipv6_cidr,
    }
    for fmt, fn in ip_generators.items():
        for _ in range(columns_per_class):
            dataset.append(_build_column(config.TYPE_IP_ADDRESS, fmt, fn, rng))

    for fmt in config.DATE_FORMATS:
        for _ in range(columns_per_class):
            ambiguity = rng.choice(["none", "resolvable", "unresolvable"])
            dataset.append(_build_date_column(fmt, rng, ambiguity))

    for fmt in config.DATE_TIME_FORMATS:
        for _ in range(columns_per_class):
            dataset.append(_build_datetime_column(fmt, rng))

    for fmt in config.TIME_FORMATS:
        for _ in range(columns_per_class):
            dataset.append(_build_time_column(fmt, rng))

    for _ in range(columns_per_class):
        dataset.append(_build_not_applicable_column(rng))

    n_ambiguous = int(columns_per_class * AMBIGUOUS_FRACTION)

    def _binary_column(type_name: str, neutral_name: bool) -> dict:
        n = rng.randint(MIN_COL_LEN, MAX_COL_LEN)
        values = [str(rng.choice([0, 0, 1, 1])) for _ in range(n)]
        values = _maybe_inject_nulls(values, rng)
        name = rng.choice(NEUTRAL_NAMES) if neutral_name else rng.choice(NAME_HINTS[type_name])
        return {
            "name": name,
            "values": values,
            "type": type_name,
            "format": None,
            "ambiguity": "none",
        }

    for i in range(n_ambiguous):
        neutral = i % 2 == 0
        dataset.append(_binary_column(config.TYPE_WHOLE_NUMBER, neutral))
        dataset.append(_binary_column(config.TYPE_TRUE_FALSE, neutral))

    rng.shuffle(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labelled columns.")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "labeled_columns.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=COLUMNS_PER_CLASS,
        help="Columns to generate per (type/format) class.",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    dataset = generate_dataset(columns_per_class=args.per_class, seed=args.seed)

    with open(args.out, "w") as f:
        json.dump(dataset, f)

    from collections import Counter

    type_counts = Counter(c["type"] for c in dataset)
    ambig_counts = Counter(
        c["ambiguity"] for c in dataset if c["type"] == config.TYPE_DATE
    )
    print(f"Generated {len(dataset)} labelled columns -> {args.out}\n")
    print("By type:")
    for t, n in sorted(type_counts.items()):
        print(f"  {t:<16} {n}")
    print("\nDate ambiguity breakdown:")
    for a, n in sorted(ambig_counts.items()):
        print(f"  {a:<14} {n}")


if __name__ == "__main__":
    main()