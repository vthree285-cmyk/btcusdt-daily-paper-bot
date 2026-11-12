
import json
import os
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

STATE_FILE = DATA / "state.json"
TRADES_FILE = DATA / "trades.csv"
SIGNALS_FILE = DATA / "signals.csv"

SYMBOL = "BTCUSDT"
INTERVAL = "1d"
FEE_PERCENT = 0.24
CHECKPOINT = "2026-11-11"


def get_candles():
    url = "https://api.binance.com/api/v3/klines"
    response = requests.get(
        url,
        params={"symbol": SYMBOL, "interval": INTERVAL, "limit": 1000},
        timeout=30,
    )
    response.raise_for_status()

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(response.json(), columns=columns)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col])

    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Ignore the currently forming daily candle.
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] < now].copy()
    return df.reset_index(drop=True)


def prepare_indicators(df):
    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr14_previous"] = true_range.rolling(14).mean().shift(1)
    df["daily_range"] = df["high"] - df["low"]
    df["range_ratio"] = df["daily_range"] / df["atr14_previous"]

    # Threshold uses only the preceding 365 days.
    df["range_threshold"] = (
        df["range_ratio"]
        .rolling(365, min_periods=365)
        .quantile(0.667)
        .shift(1)
    )

    candle_range = df["high"] - df["low"]
    df["close_percentile"] = (
        100 * (df["close"] - df["low"]) / candle_range
    )
    return df


def read_csv(path, columns):
    if path.exists() and path.stat().st_size:
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def main():
    state = {"last_processed_signal_date": CHECKPOINT}
    if STATE_FILE.exists():
        state.update(json.loads(STATE_FILE.read_text()))

    last_date = state["last_processed_signal_date"]

    df = prepare_indicators(get_candles())

    trade_columns = [
        "signal_date", "entry_date", "entry_price", "exit_price",
        "gross_return_pct", "cost_pct", "net_return_pct", "result",
    ]
    signal_columns = [
        "signal_date", "close_percentile", "range_ratio",
        "range_threshold",
    ]

    trades = read_csv(TRADES_FILE, trade_columns)
    signals = read_csv(SIGNALS_FILE, signal_columns)

    existing_signals = set(signals.get("signal_date", pd.Series(dtype=str)).astype(str))
    existing_trades = set(trades.get("signal_date", pd.Series(dtype=str)).astype(str))

    # A signal is evaluated only when its following day's candle has closed.
    new_checkpoint = last_date

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        signal_date = str(row["date"])

        if signal_date <= last_date:
            continue

        if pd.isna(row["close_percentile"]) or pd.isna(row["range_threshold"]):
            new_checkpoint = signal_date
            continue

        qualifies = (
            row["close_percentile"] <= 10
            and row["range_ratio"] > row["range_threshold"]
        )

        if qualifies:
            if signal_date not in existing_signals:
                signals.loc[len(signals)] = {
                    "signal_date": signal_date,
                    "close_percentile": row["close_percentile"],
                    "range_ratio": row["range_ratio"],
                    "range_threshold": row["range_threshold"],
                }
                existing_signals.add(signal_date)

            if signal_date not in existing_trades:
                entry = float(next_row["open"])
                exit_price = float(next_row["close"])
                gross = (exit_price / entry - 1) * 100
                net = gross - FEE_PERCENT

                trades.loc[len(trades)] = {
                    "signal_date": signal_date,
                    "entry_date": next_row["date"],
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "gross_return_pct": gross,
                    "cost_pct": FEE_PERCENT,
                    "net_return_pct": net,
                    "result": "WIN" if net > 0 else "LOSS",
                }
                existing_trades.add(signal_date)

                print(f"Trade recorded: {signal_date} | net {net:.4f}%")

        new_checkpoint = signal_date

    trades.to_csv(TRADES_FILE, index=False)
    signals.to_csv(SIGNALS_FILE, index=False)

    state["last_processed_signal_date"] = new_checkpoint
    STATE_FILE.write_text(json.dumps(state, indent=2))

    print(f"Last processed signal date: {new_checkpoint}")
    print(f"Total paper trades: {len(trades)}")


if __name__ == "__main__":
    main()
