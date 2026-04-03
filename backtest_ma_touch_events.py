from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from long_term_screener import calc_indicators
from market_data_utils import prepare_price_history


DETAILS_CSV = "ma_touch_event_details.csv"
SUMMARY_CSV = "ma_touch_event_summary.csv"
FORWARD_WINDOWS = (5, 10, 20, 60)
DEFAULT_EVENT_COOLDOWN_DAYS = 20
DEFAULT_MAX_ABS_RETURN_PCT = 300.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest MA touch events for medium-term screening.")
    parser.add_argument("--tickers-csv", default="tickers.csv", help="Path to ticker universe CSV.")
    parser.add_argument(
        "--output-dir",
        default="results/ma_touch_backtests",
        help="Directory for event details and summary CSVs.",
    )
    parser.add_argument("--period", default="5y", help="yfinance history period. Default: 5y")
    parser.add_argument("--min-history", type=int, default=260, help="Minimum required rows per ticker.")
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional limit for number of tickers.")
    parser.add_argument("--offset", type=int, default=0, help="Start offset in ticker universe.")
    parser.add_argument("--limit", type=int, default=0, help="Optional slice size from offset (0 = no limit).")
    parser.add_argument(
        "--event-cooldown-days",
        type=int,
        default=DEFAULT_EVENT_COOLDOWN_DAYS,
        help="Minimum spacing between events for the same ticker/touch type.",
    )
    parser.add_argument(
        "--max-abs-return-pct",
        type=float,
        default=DEFAULT_MAX_ABS_RETURN_PCT,
        help="Ignore forward returns and runups/drawdowns whose absolute value exceeds this threshold.",
    )
    return parser.parse_args()


def load_tickers(path: Path, max_tickers: int, offset: int, limit: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["ticker"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    offset = max(offset, 0)
    if limit and limit > 0:
        df = df.iloc[offset : offset + limit]
    elif max_tickers and max_tickers > 0:
        df = df.iloc[offset : offset + max_tickers]
    elif offset > 0:
        df = df.iloc[offset:]
    return df.reset_index(drop=True)


def fetch_history(ticker: str, period: str) -> pd.DataFrame | None:
    obj = yf.Ticker(ticker)
    try:
        hist = obj.history(period=period, interval="1d", auto_adjust=False, actions=True)
    except Exception as exc:
        print(f"history error {ticker}: {exc}")
        return None
    return prepare_price_history(hist)


def _sanitize_return(value: float, max_abs_return_pct: float) -> float | None:
    if not pd.notna(value):
        return None
    if abs(value) > max_abs_return_pct:
        return None
    return round(value, 3)


def _future_return(df: pd.DataFrame, idx: int, horizon: int, max_abs_return_pct: float) -> float | None:
    target_idx = idx + horizon
    if target_idx >= len(df):
        return None
    entry = float(df.iloc[idx]["Close"])
    exit_ = float(df.iloc[target_idx]["Close"])
    if entry <= 0:
        return None
    raw = (exit_ - entry) / entry * 100
    return _sanitize_return(raw, max_abs_return_pct)


def _max_runup_drawdown(
    df: pd.DataFrame, idx: int, horizon: int, max_abs_return_pct: float
) -> tuple[float | None, float | None]:
    end_idx = min(idx + horizon, len(df) - 1)
    if idx + 1 > end_idx:
        return None, None
    entry = float(df.iloc[idx]["Close"])
    if entry <= 0:
        return None, None
    window = df.iloc[idx + 1 : end_idx + 1]
    if window.empty:
        return None, None
    max_high = float(window["High"].max())
    min_low = float(window["Low"].min())
    runup = _sanitize_return((max_high - entry) / entry * 100, max_abs_return_pct)
    drawdown = _sanitize_return((min_low - entry) / entry * 100, max_abs_return_pct)
    return runup, drawdown


def _stop_hit(df: pd.DataFrame, idx: int, horizon: int, stop_price: float | None) -> bool | None:
    if stop_price is None:
        return None
    end_idx = min(idx + horizon, len(df) - 1)
    if idx + 1 > end_idx:
        return None
    window = df.iloc[idx + 1 : end_idx + 1]
    if window.empty:
        return None
    return bool((window["Low"] <= stop_price).any())


def _event_record(
    df: pd.DataFrame,
    row_idx: int,
    ticker: str,
    name: str,
    touch_type: str,
    entry_rule: str,
    entry_idx: int,
    stop_price: float | None,
    max_abs_return_pct: float,
) -> dict:
    signal_row = df.iloc[row_idx]
    entry_row = df.iloc[entry_idx]
    record = {
        "ticker": ticker,
        "name": name,
        "signal_date": str(pd.Timestamp(signal_row.name).date()),
        "entry_date": str(pd.Timestamp(entry_row.name).date()),
        "touch_type": touch_type,
        "entry_rule": entry_rule,
        "signal_close": round(float(signal_row["Close"]), 3),
        "entry_price": round(float(entry_row["Close"]), 3),
        "ma25": round(float(signal_row["ma25"]), 3) if pd.notna(signal_row["ma25"]) else None,
        "ma75": round(float(signal_row["ma75"]), 3) if pd.notna(signal_row["ma75"]) else None,
        "ma200": round(float(signal_row["ma200"]), 3) if pd.notna(signal_row["ma200"]) else None,
        "touch_ma25_intraday": bool(signal_row["touch_ma25_intraday"]),
        "touch_ma75_intraday": bool(signal_row["touch_ma75_intraday"]),
        "reclaim_ma25_close": bool(signal_row["reclaim_ma25_close"]),
        "reclaim_ma75_close": bool(signal_row["reclaim_ma75_close"]),
        "support_reaction_ok": bool(signal_row.get("support_reaction_ok", False)),
        "ma25_slope_pct": round(float(signal_row["ma25_slope_pct"]), 3) if pd.notna(signal_row["ma25_slope_pct"]) else None,
        "ma75_slope_pct": round(float(signal_row["ma75_slope_pct"]), 3) if pd.notna(signal_row["ma75_slope_pct"]) else None,
        "ma200_slope_pct": round(float(signal_row["ma200_slope_pct"]), 3) if pd.notna(signal_row["ma200_slope_pct"]) else None,
        "close_vs_ma25_pct": round(float(signal_row["close_vs_ma25_pct"]), 3) if pd.notna(signal_row["close_vs_ma25_pct"]) else None,
        "close_vs_ma75_pct": round(float(signal_row["close_vs_ma75_pct"]), 3) if pd.notna(signal_row["close_vs_ma75_pct"]) else None,
        "drawdown_from_60d_high_pct": round(float(signal_row["drawdown_from_60d_high_pct"]), 3)
        if pd.notna(signal_row["drawdown_from_60d_high_pct"])
        else None,
        "volume_ratio_20": round(float(signal_row["volume_ratio_20"]), 3) if pd.notna(signal_row["volume_ratio_20"]) else None,
        "upper_shadow_pct": round(float(signal_row["upper_shadow_pct"]), 3) if pd.notna(signal_row["upper_shadow_pct"]) else None,
        "lower_shadow_pct": round(float(signal_row["lower_shadow_pct"]), 3) if pd.notna(signal_row["lower_shadow_pct"]) else None,
        "stop_price": round(float(stop_price), 3) if stop_price is not None else None,
    }

    for horizon in FORWARD_WINDOWS:
        record[f"ret_{horizon}d_pct"] = _future_return(df, entry_idx, horizon, max_abs_return_pct)

    runup_20, drawdown_20 = _max_runup_drawdown(df, entry_idx, 20, max_abs_return_pct)
    record["max_runup_20d_pct"] = runup_20
    record["max_drawdown_20d_pct"] = drawdown_20
    record["stop_hit_5d"] = _stop_hit(df, entry_idx, 5, stop_price)
    record["stop_hit_20d"] = _stop_hit(df, entry_idx, 20, stop_price)
    return record


def build_events(
    df: pd.DataFrame,
    ticker: str,
    name: str,
    event_cooldown_days: int,
    max_abs_return_pct: float,
) -> list[dict]:
    events: list[dict] = []
    last_event_idx: dict[str, int] = {}

    for idx in range(200, len(df) - max(FORWARD_WINDOWS)):
        row = df.iloc[idx]
        touch_types = []
        if bool(row["touch_ma25_intraday"]):
            touch_types.append(("ma25", float(row["ma25"]) if pd.notna(row["ma25"]) else None))
        if bool(row["touch_ma75_intraday"]):
            touch_types.append(("ma75", float(row["ma75"]) if pd.notna(row["ma75"]) else None))

        if not touch_types:
            continue

        next_idx = idx + 1
        if next_idx >= len(df):
            continue
        next_row = df.iloc[next_idx]
        confirm_next_close = bool(next_row["Close"] > next_row["Open"])
        confirm_high_break = bool(next_row["High"] > row["High"])

        for touch_type, stop_ref in touch_types:
            previous_idx = last_event_idx.get(touch_type)
            if previous_idx is not None and idx - previous_idx < event_cooldown_days:
                continue

            stop_price = min(float(row["Low"]), stop_ref) if stop_ref is not None else float(row["Low"])
            events.append(
                _event_record(df, idx, ticker, name, touch_type, "touch_close", idx, stop_price, max_abs_return_pct)
            )
            events.append(
                _event_record(
                    df, idx, ticker, name, touch_type, "next_close", next_idx, stop_price, max_abs_return_pct
                )
            )

            if touch_type == "ma25":
                signal_reclaim = bool(row["reclaim_ma25_close"])
                next_close_above_ma = bool(next_row["Close"] >= next_row["ma25"])
            else:
                signal_reclaim = bool(row["reclaim_ma75_close"])
                next_close_above_ma = bool(next_row["Close"] >= next_row["ma75"])

            lower_shadow_pct = float(row["lower_shadow_pct"]) if pd.notna(row["lower_shadow_pct"]) else 0.0
            volume_ratio_20 = float(row["volume_ratio_20"]) if pd.notna(row["volume_ratio_20"]) else 0.0
            signal_absorption = bool(row.get("support_reaction_ok", False)) or lower_shadow_pct >= 1.0 or volume_ratio_20 >= 1.2
            next_hold_low = bool(next_row["Low"] >= row["Low"])
            next_follow_through = confirm_next_close and bool(next_row["Close"] >= row["Close"]) and next_hold_low

            confirmed = (signal_reclaim and signal_absorption and next_follow_through) or (
                confirm_high_break and next_follow_through and next_close_above_ma
            )

            if confirmed:
                events.append(
                    _event_record(
                        df,
                        idx,
                        ticker,
                        name,
                        touch_type,
                        "confirm_close",
                        next_idx,
                        stop_price,
                        max_abs_return_pct,
                    )
                )

            last_event_idx[touch_type] = idx

    return events


def summarize_events(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    rows = []
    grouped = events_df.groupby(["touch_type", "entry_rule"], dropna=False)
    for (touch_type, entry_rule), group in grouped:
        row = {
            "touch_type": touch_type,
            "entry_rule": entry_rule,
            "count": int(len(group)),
        }
        for horizon in FORWARD_WINDOWS:
            col = f"ret_{horizon}d_pct"
            valid = group[col].dropna()
            row[f"win_rate_{horizon}d_pct"] = round((valid > 0).mean() * 100, 2) if not valid.empty else None
            row[f"avg_{horizon}d_pct"] = round(valid.mean(), 3) if not valid.empty else None
            row[f"median_{horizon}d_pct"] = round(valid.median(), 3) if not valid.empty else None

        drawdown = group["max_drawdown_20d_pct"].dropna()
        runup = group["max_runup_20d_pct"].dropna()
        row["avg_max_drawdown_20d_pct"] = round(drawdown.mean(), 3) if not drawdown.empty else None
        row["avg_max_runup_20d_pct"] = round(runup.mean(), 3) if not runup.empty else None

        stop_hit_5d = group["stop_hit_5d"].dropna()
        stop_hit_20d = group["stop_hit_20d"].dropna()
        row["stop_hit_5d_rate_pct"] = round(stop_hit_5d.mean() * 100, 2) if not stop_hit_5d.empty else None
        row["stop_hit_20d_rate_pct"] = round(stop_hit_20d.mean() * 100, 2) if not stop_hit_20d.empty else None
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["touch_type", "entry_rule"]).reset_index(drop=True)


def run() -> None:
    args = parse_args()
    tickers_path = Path(args.tickers_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers_df = load_tickers(tickers_path, args.max_tickers, args.offset, args.limit)
    all_events: list[dict] = []

    slice_label = f"offset{args.offset}"
    if args.limit and args.limit > 0:
        slice_label += f"_limit{args.limit}"
    elif args.max_tickers and args.max_tickers > 0:
        slice_label += f"_limit{args.max_tickers}"

    for i, row in tickers_df.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        print(f"[{i + 1}/{len(tickers_df)}] backtesting {ticker} {name}")
        hist = fetch_history(ticker, args.period)
        if hist is None or hist.empty or len(hist) < args.min_history:
            continue

        enriched = calc_indicators(hist)
        events = build_events(
            enriched,
            ticker,
            name,
            event_cooldown_days=max(args.event_cooldown_days, 1),
            max_abs_return_pct=max(args.max_abs_return_pct, 1.0),
        )
        all_events.extend(events)

    events_df = pd.DataFrame(all_events)
    details_path = output_dir / DETAILS_CSV
    summary_path = output_dir / SUMMARY_CSV
    if args.offset > 0 or (args.limit and args.limit > 0) or (args.max_tickers and args.max_tickers > 0):
        details_path = output_dir / f"ma_touch_event_details_{slice_label}.csv"
        summary_path = output_dir / f"ma_touch_event_summary_{slice_label}.csv"
    events_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    summary_df = summarize_events(events_df)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"saved details: {details_path}")
    print(f"saved summary: {summary_path}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run()
