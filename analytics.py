"""Performance analytics over the trade ledger.

Win rate alone tells you almost nothing — a 70% win rate with a bad
reward:risk still loses money. This module computes the metrics that actually
decide whether the strategy has an edge after costs:

  - net PnL, return on initial equity
  - profit factor (gross wins / gross losses) — needs > 1.0, ideally > 1.5
  - expectancy per trade (average net PnL) — must exceed 0 after fees
  - average win / average loss and realized reward:risk
  - max drawdown from the equity curve — the number that kills accounts
  - per-trade Sharpe (mean / std of net PnL)
  - longest losing streak — what your psychology must survive
  - breakdown by exit reason and by symbol

Usage:
    python analytics.py                    # reads DB_PATH from .env
    python analytics.py backtest.db        # explicit ledger
    python analytics.py backtest.db --csv  # also export trades + equity CSVs
"""

from __future__ import annotations

import math
import sqlite3
import sys


def _rows(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(sql, args).fetchall()


def compute_metrics(db_path: str, initial_equity: float | None = None) -> dict:
    conn = sqlite3.connect(db_path)
    trades = _rows(conn, "SELECT * FROM trades ORDER BY closed_ts ASC")
    curve = _rows(conn, "SELECT * FROM equity_curve ORDER BY ts ASC")
    conn.close()

    pnls = [float(t["net_pnl"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    expectancy = (sum(pnls) / len(pnls)) if pnls else 0.0
    avg_win = (gross_win / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    realized_rr = (avg_win / avg_loss) if avg_loss > 0 else float("inf") if avg_win > 0 else 0.0

    # Per-trade Sharpe (not annualized; comparable across runs of same timeframe).
    sharpe = 0.0
    if len(pnls) > 1:
        mean = expectancy
        var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(var)
        sharpe = mean / std if std > 0 else 0.0

    # Max drawdown over the equity curve.
    max_dd = 0.0
    max_dd_pct = 0.0
    peak = -float("inf")
    for row in curve:
        eq = float(row["equity"])
        peak = max(peak, eq)
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak if peak > 0 else 0.0

    # Longest losing streak.
    longest_streak = streak = 0
    for p in pnls:
        streak = streak + 1 if p <= 0 else 0
        longest_streak = max(longest_streak, streak)

    # Breakdown by exit reason and symbol.
    by_reason: dict[str, dict] = {}
    by_symbol: dict[str, dict] = {}
    for t in trades:
        for key, bucket in ((t["reason"], by_reason), (t["symbol"], by_symbol)):
            b = bucket.setdefault(key, {"n": 0, "pnl": 0.0})
            b["n"] += 1
            b["pnl"] += float(t["net_pnl"])

    fees_total = sum(float(t["fees"]) for t in trades)
    start_eq = initial_equity if initial_equity is not None else (
        float(curve[0]["equity"]) - pnls[0] if curve and pnls else None
    )
    end_eq = float(curve[-1]["equity"]) if curve else None

    return {
        "trades": len(pnls),
        "win_rate": (len(wins) / len(pnls)) if pnls else 0.0,
        "net_pnl": sum(pnls),
        "fees_total": fees_total,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "realized_rr": realized_rr,
        "sharpe_per_trade": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "longest_losing_streak": longest_streak,
        "by_reason": by_reason,
        "by_symbol": by_symbol,
        "start_equity": start_eq,
        "end_equity": end_eq,
    }


def print_report(m: dict) -> None:
    print("=" * 58)
    print("PERFORMANCE REPORT")
    print("=" * 58)
    if m["trades"] == 0:
        print("No closed trades in this ledger yet.")
        return
    if m["start_equity"] is not None and m["end_equity"] is not None and m["start_equity"] > 0:
        ret = (m["end_equity"] - m["start_equity"]) / m["start_equity"] * 100
        print(f"Equity            {m['start_equity']:.4f} -> {m['end_equity']:.4f}  ({ret:+.2f}%)")
    print(f"Trades            {m['trades']}   win rate {m['win_rate'] * 100:.1f}%")
    print(f"Net PnL           {m['net_pnl']:+.4f}   (fees paid {m['fees_total']:.4f})")
    print(f"Profit factor     {m['profit_factor']:.2f}   (need > 1.0, healthy > 1.5)")
    print(f"Expectancy/trade  {m['expectancy']:+.5f}")
    print(f"Avg win / loss    {m['avg_win']:.4f} / {m['avg_loss']:.4f}   realized R:R {m['realized_rr']:.2f}")
    print(f"Sharpe (trade)    {m['sharpe_per_trade']:.2f}")
    print(f"Max drawdown      {m['max_drawdown']:.4f}  ({m['max_drawdown_pct'] * 100:.1f}% of peak)")
    print(f"Longest L-streak  {m['longest_losing_streak']}")
    print("-" * 58)
    print("By exit reason:")
    for reason, b in sorted(m["by_reason"].items()):
        print(f"  {reason:<14} n={b['n']:<4} pnl={b['pnl']:+.4f}")
    print("By symbol:")
    for sym, b in sorted(m["by_symbol"].items()):
        print(f"  {sym:<14} n={b['n']:<4} pnl={b['pnl']:+.4f}")
    print("=" * 58)
    verdict = "NET POSITIVE after costs" if m["net_pnl"] > 0 and m["profit_factor"] > 1.0 else \
        "NOT profitable after costs — do not go live"
    print(f"Verdict: {verdict}")


def export_csv(db_path: str) -> None:
    import csv

    conn = sqlite3.connect(db_path)
    for table, fname in (("trades", "trades.csv"), ("equity_curve", "equity_curve.csv")):
        rows = _rows(conn, f"SELECT * FROM {table}")
        if not rows:
            continue
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rows[0].keys())
            writer.writerows([tuple(r) for r in rows])
        print(f"Exported {len(rows)} rows -> {fname}")
    conn.close()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    db = args[0] if args else None
    if db is None:
        from config.settings import load_config

        db = load_config().db_path
    metrics = compute_metrics(db)
    print_report(metrics)
    if "--csv" in sys.argv:
        export_csv(db)
