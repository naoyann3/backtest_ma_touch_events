from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yfinance as yf

from market_data_utils import prepare_price_history, select_latest_completed_row


DETAILS_CSV = "sector_return_details.csv"
SUMMARY_CSV = "sector_return_summary.csv"
RETURN_WINDOWS = {
    "1y": 252,
    "6m": 126,
    "3m": 63,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare basket performance for a sector/theme.")
    parser.add_argument("--tickers-csv", default="tickers.csv", help="Path to ticker universe CSV.")
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated tickers to compare. Example: 6857.T,6146.T,7735.T",
    )
    parser.add_argument(
        "--label",
        default="custom_sector",
        help="Sector/theme label for output summary.",
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="yfinance period to download. Use at least 2y if you want stable 1y comparisons.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/sector_returns",
        help="Directory for details and summary CSVs.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=260,
        help="Minimum required rows per ticker.",
    )
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip())
    return cleaned.strip("_") or "custom_sector"


def parse_ticker_list(raw: str) -> list[str]:
    tickers = [item.strip() for item in raw.split(",") if item.strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            ordered.append(ticker)
    return ordered


def load_ticker_names(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return {}
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return dict(zip(df["ticker"], df["name"].astype(str)))


def fetch_history(ticker: str, period: str) -> pd.DataFrame | None:
    obj = yf.Ticker(ticker)
    try:
        hist = obj.history(period=period, interval="1d", auto_adjust=False, actions=True)
    except Exception as exc:
        print(f"history error {ticker}: {exc}")
        return None
    return prepare_price_history(hist)


def _return_from_offset(df: pd.DataFrame, latest_idx: int, offset: int) -> float | None:
    target_idx = latest_idx - offset
    if target_idx < 0:
        return None
    start_price = float(df.iloc[target_idx]["Close"])
    end_price = float(df.iloc[latest_idx]["Close"])
    if start_price <= 0:
        return None
    return round((end_price - start_price) / start_price * 100, 3)


def build_details(
    tickers: list[str],
    ticker_names: dict[str, str],
    period: str,
    min_history: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx}/{len(tickers)}] sector compare {ticker}")
        hist = fetch_history(ticker, period)
        if hist is None or len(hist) < min_history:
            print(f"skip {ticker}: insufficient history")
            continue

        latest_row, latest_date = select_latest_completed_row(hist)
        latest_idx = hist.index.get_loc(latest_row.name)
        row = {
            "ticker": ticker,
            "name": ticker_names.get(ticker, ticker),
            "latest_date": str(latest_date),
            "close": round(float(latest_row["raw_close"]), 3),
            "rows": int(len(hist)),
        }
        for label, offset in RETURN_WINDOWS.items():
            row[f"ret_{label}_pct"] = _return_from_offset(hist, latest_idx, offset)
        rows.append(row)

    return pd.DataFrame(rows)


def _summary_value(series: pd.Series, fn: str) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    if fn == "mean":
        return round(float(clean.mean()), 3)
    if fn == "median":
        return round(float(clean.median()), 3)
    if fn == "positive_ratio":
        return round(float((clean > 0).mean() * 100), 3)
    if fn == "count":
        return int(clean.count())
    raise ValueError(fn)


def build_summary(details_df: pd.DataFrame, label: str) -> pd.DataFrame:
    if details_df.empty:
        return pd.DataFrame(
            [
                {
                    "sector_label": label,
                    "ticker_count": 0,
                    "latest_date": None,
                }
            ]
        )

    latest_date = details_df["latest_date"].max()
    summary: dict[str, object] = {
        "sector_label": label,
        "ticker_count": int(len(details_df)),
        "latest_date": latest_date,
    }

    for horizon in RETURN_WINDOWS:
        col = f"ret_{horizon}_pct"
        summary[f"avg_{horizon}_pct"] = _summary_value(details_df[col], "mean")
        summary[f"median_{horizon}_pct"] = _summary_value(details_df[col], "median")
        summary[f"positive_ratio_{horizon}_pct"] = _summary_value(details_df[col], "positive_ratio")
        summary[f"count_{horizon}"] = _summary_value(details_df[col], "count")

        clean = pd.to_numeric(details_df[col], errors="coerce").dropna()
        if clean.empty:
            summary[f"top_ticker_{horizon}"] = None
            summary[f"top_return_{horizon}_pct"] = None
            summary[f"bottom_ticker_{horizon}"] = None
            summary[f"bottom_return_{horizon}_pct"] = None
        else:
            top_idx = clean.idxmax()
            bottom_idx = clean.idxmin()
            summary[f"top_ticker_{horizon}"] = str(details_df.loc[top_idx, "ticker"])
            summary[f"top_return_{horizon}_pct"] = round(float(clean.loc[top_idx]), 3)
            summary[f"bottom_ticker_{horizon}"] = str(details_df.loc[bottom_idx, "ticker"])
            summary[f"bottom_return_{horizon}_pct"] = round(float(clean.loc[bottom_idx]), 3)

    return pd.DataFrame([summary])


def main() -> None:
    args = parse_args()
    tickers = parse_ticker_list(args.tickers)
    if not tickers:
        raise SystemExit("No tickers provided. Pass --tickers.")

    label = sanitize_label(args.label)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ticker_names = load_ticker_names(Path(args.tickers_csv))
    missing = [ticker for ticker in tickers if ticker not in ticker_names]
    if missing:
        print(f"warning: {len(missing)} tickers not found in universe CSV: {', '.join(missing)}")

    details_df = build_details(
        tickers=tickers,
        ticker_names=ticker_names,
        period=args.period,
        min_history=args.min_history,
    )
    summary_df = build_summary(details_df, label)

    details_path = output_dir / f"{label}_{DETAILS_CSV}"
    summary_path = output_dir / f"{label}_{SUMMARY_CSV}"
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"details_rows={len(details_df)}")
    print(f"summary_rows={len(summary_df)}")
    print(f"wrote {details_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
