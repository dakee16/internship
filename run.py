from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.type_format_suggester import suggest  # noqa: E402


def analyse_csv(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"{len(df)} rows × {len(df.columns)} columns\n")

    rows = []
    for col_name in df.columns:
        series = df[col_name].copy()
        series.name = col_name

        result = suggest(series)

        sample = (
            series.dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .head(3)
            .tolist()
        )
        sample_str = " | ".join(str(v) for v in sample)

        null_pct = round(series.isna().mean() * 100, 1)

        rows.append({
            "column_name":    col_name,
            "suggested_type": result["type"],
            "format":         result["format"] if result["format"] else "",
            "confidence":     result["confidence"],
            "sample_values":  sample_str,
            "null_percent":   null_pct,
        })

        fmt_str = f'  [{result["format"]}]' if result["format"] else ""
        print(
            f"  {col_name:<30} →  {result['type']:<16}{fmt_str:<18}"
            f"  conf={result['confidence']:.2f}  nulls={null_pct}%"
        )

    results_df = (
        pd.DataFrame(rows)
        .sort_values("confidence", ascending=False)
        .reset_index(drop=True)
    )

    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved → {output_path}")
    print(f"    {len(results_df)} columns analysed\n")

    print("── Type breakdown ──────────────────────")
    type_counts = results_df["suggested_type"].value_counts()
    for t, n in type_counts.items():
        print(f"  {t:<18} {n}")

    low_conf = results_df[results_df["confidence"] < 0.80]
    if len(low_conf):
        print(f"\n{len(low_conf)} column(s) with confidence < 0.80 "
              f"(may need manual review):")
        for _, row in low_conf.iterrows():
            print(f"  {row['column_name']:<30}  conf={row['confidence']:.2f}  "
                  f"type={row['suggested_type']}")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse a CSV and suggest a semantic type for every column."
    )
    parser.add_argument("input", help="Path to the input CSV file.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <input>_typed.csv next to the input file).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        sys.exit(1)

    if args.out:
        output_path = args.out
    else:
        base, _ = os.path.splitext(args.input)
        output_path = base + "_typed.csv"

    analyse_csv(args.input, output_path)


if __name__ == "__main__":
    main()