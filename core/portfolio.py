"""Position state and simulated PnL ledger (SQLite).

This module is the answer to two of the original code's worst bugs:

  1. It holds open-position state, so the agent never re-enters a symbol it is
     already in (no order stacking every cycle).
  2. In DRY_RUN it simulates fills, fees, slippage, and stop/target exits, and
     tracks running equity, so a strategy can actually be proven net-positive
     before any real money is committed.

The same position table is used in LIVE mode to remember what is open between
loop iterations.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from config.settings import Config
from core.risk import Bracket

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    side: str          # "buy" (long) or "sell" (short)
    entry: float
    stop_loss: float
    take_profit: float
    quantity: float
    opened_ts: int


class Portfolio:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.conn = sqlite3.connect(config.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._equity = self._load_equity()

    # --- schema ---------------------------------------------------------
    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                side TEXT, entry REAL, stop_loss REAL, take_profit REAL,
                quantity REAL, opened_ts INTEGER
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, side TEXT, entry REAL, exit REAL,
                quantity REAL, gross_pnl REAL, fees REAL, net_pnl REAL,
                reason TEXT, opened_ts INTEGER, closed_ts INTEGER, mode TEXT
            );
            CREATE TABLE IF NOT EXISTS equity_curve (
                ts INTEGER, equity REAL
            );
            """
        )
        self.conn.commit()

    def _load_equity(self) -> float:
        row = self.conn.execute("SELECT equity FROM equity_curve ORDER BY ts DESC LIMIT 1").fetchone()
        return float(row["equity"]) if row else self.config.initial_equity

    # --- position state -------------------------------------------------
    def get_position(self, symbol: str) -> Position | None:
        row = self.conn.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,)).fetchone()
        if not row:
            return None
        return Position(
            symbol=row["symbol"], side=row["side"], entry=row["entry"],
            stop_loss=row["stop_loss"], take_profit=row["take_profit"],
            quantity=row["quantity"], opened_ts=row["opened_ts"],
        )

    def has_position(self, symbol: str) -> bool:
        return self.get_position(symbol) is not None

    def open_position(self, bracket: Bracket, symbol: str, ts: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?)",
            (symbol, bracket.side, bracket.entry, bracket.stop_loss,
             bracket.take_profit, bracket.quantity, ts),
        )
        self.conn.commit()
        logger.info("Opened %s %s qty=%.6f @ %.4f (SL %.4f / TP %.4f)",
                    bracket.side, symbol, bracket.quantity, bracket.entry,
                    bracket.stop_loss, bracket.take_profit)

    def close_position(self, symbol: str, exit_price: float, reason: str, ts: int) -> float:
        """Close a position, record the trade, update equity. Returns net PnL."""
        pos = self.get_position(symbol)
        if pos is None:
            return 0.0

        direction = 1 if pos.side == "buy" else -1
        gross = (exit_price - pos.entry) * pos.quantity * direction
        # Taker fee + slippage charged on both entry and exit notionals.
        cost_rate = self.config.taker_fee + self.config.slippage
        fees = (pos.entry + exit_price) * pos.quantity * cost_rate
        net = gross - fees

        self._equity += net
        self.conn.execute(
            "INSERT INTO trades (symbol, side, entry, exit, quantity, gross_pnl, "
            "fees, net_pnl, reason, opened_ts, closed_ts, mode) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (symbol, pos.side, pos.entry, exit_price, pos.quantity, gross, fees,
             net, reason, pos.opened_ts, ts, self.config.trade_mode),
        )
        self.conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        self.conn.execute("INSERT INTO equity_curve VALUES (?,?)", (ts, self._equity))
        self.conn.commit()
        logger.info("Closed %s @ %.4f (%s) net=%.4f equity=%.4f",
                    symbol, exit_price, reason, net, self._equity)
        return net

    # --- simulation -----------------------------------------------------
    def check_exits(self, symbol: str, candle_high: float, candle_low: float, ts: int) -> str | None:
        """In DRY_RUN, resolve stop/target hits against a candle's range.

        If both levels fall inside the candle we conservatively assume the stop
        filled first (worst case), which keeps simulated results honest.
        """
        pos = self.get_position(symbol)
        if pos is None:
            return None

        if pos.side == "buy":
            stop_hit = candle_low <= pos.stop_loss
            tp_hit = candle_high >= pos.take_profit
            if stop_hit:
                self.close_position(symbol, pos.stop_loss, "STOP_LOSS", ts)
                return "STOP_LOSS"
            if tp_hit:
                self.close_position(symbol, pos.take_profit, "TAKE_PROFIT", ts)
                return "TAKE_PROFIT"
        else:  # short
            stop_hit = candle_high >= pos.stop_loss
            tp_hit = candle_low <= pos.take_profit
            if stop_hit:
                self.close_position(symbol, pos.stop_loss, "STOP_LOSS", ts)
                return "STOP_LOSS"
            if tp_hit:
                self.close_position(symbol, pos.take_profit, "TAKE_PROFIT", ts)
                return "TAKE_PROFIT"
        return None

    @property
    def equity(self) -> float:
        return self._equity

    def stats(self) -> dict:
        rows = self.conn.execute("SELECT net_pnl FROM trades").fetchall()
        pnls = [r["net_pnl"] for r in rows]
        wins = [p for p in pnls if p > 0]
        return {
            "trades": len(pnls),
            "wins": len(wins),
            "win_rate": (len(wins) / len(pnls)) if pnls else 0.0,
            "net_pnl": sum(pnls),
            "equity": self._equity,
        }

    def close(self) -> None:
        self.conn.close()
