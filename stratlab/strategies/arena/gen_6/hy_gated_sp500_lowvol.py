"""HY-credit-gated SP500 low-volatility factor strategy.

Hypothesis:
  Dual gate: JNK above 50d SMA (credit risk-on) AND SPY above 200d SMA
  (equity bull market). When both conditions met, hold top-20 SP500 stocks
  by LOWEST 63-day realized volatility. When either signal is bearish,
  rotate to TLT. Biweekly rebalance (10 bars).

Rationale:
  Low-volatility factor (Ang et al. 2006): low-vol stocks outperform
  high-vol stocks on risk-adjusted basis due to leverage constraints and
  lottery-ticket preference. Adding a credit gate (JNK) improves timing
  — during credit stress, even low-vol stocks can fall sharply. The JNK
  signal is separate from equity price momentum, providing different
  switching dynamics.

  Key structural difference from existing strategies:
  1. Ranking criterion: LOWEST vol (not highest momentum) — inverted signal
  2. Gate: JNK 50d SMA + SPY 200d SMA (dual, vs single VIX threshold)
  3. Defensive: TLT only (simple, not mixed bond allocation)

  This should have low correlation to momentum-ranked strategies because
  the stocks selected are often different (stable, low-beta names vs
  high-momentum names are largely non-overlapping).

Diversification vs leaderboard:
  - gen6_lowbeta_momentum_sp500 (corr 0.52): uses beta + momentum filters,
    this uses vol ONLY as the ranking criterion.
  - VIX-gated strategies: VIX gate, this uses JNK credit gate.
  - gen5_xlf_kre_bank_spread (gen6 low-vol): SPY 200d only; this adds JNK.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

JNK_MA = 50          # JNK MA for credit signal
TREND_WINDOW = 200   # SPY 200d SMA
VOL_WINDOW = 63      # realized vol window for ranking
REBALANCE_EVERY = 10 # biweekly
TOP_K = 20
EXPOSURE = 0.97


class HyGatedSP500LowVol(Strategy):
    """Low-vol SP500 stocks gated by JNK credit signal + SPY trend."""

    def __init__(
        self,
        jnk_ma: int = JNK_MA,
        trend_window: int = TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            jnk_ma=jnk_ma,
            trend_window=trend_window,
            vol_window=vol_window,
            rebalance_every=rebalance_every,
            top_k=top_k,
            exposure=exposure,
        )
        self.jnk_ma = int(jnk_ma)
        self.trend_window = int(trend_window)
        self.vol_window = int(vol_window)
        self.rebalance_every = int(rebalance_every)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.vol_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # --- SPY 200d SMA ---
        spy_bull = False
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.trend_window + 5:
                spy_close = spy_hist["close"].dropna()
                spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                spy_bull = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # --- JNK credit trend ---
        jnk_bull = False
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma + 1:
                jnk_close = jnk_hist["close"].dropna()
                jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
                jnk_bull = float(jnk_close.iloc[-1]) > jnk_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not (spy_bull and jnk_bull):
            # Either credit or equity is bearish: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Both bullish: low-vol SP500 stocks
            need = self.vol_window + 2
            prices = ctx.closes_window(need)
            if len(prices) < self.vol_window:
                return []

            vols: dict[str, float] = {}
            etf_skip = {
                "SPY", "QQQ", "TLT", "SHY", "IEF", "GLD", "IAU", "AGG",
                "RSP", "DBC", "JNK", "LQD", "HYG", "SSO", "TQQQ",
                "XLK", "XLV", "XLF", "XLI", "XLP", "XLU", "XLE", "XLB",
                "XLRE", "XLY", "XLC", "MTUM", "VLUE", "VTV", "VUG",
            }
            for sym in prices.columns:
                if sym in etf_skip:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.vol_window + 1:
                    continue
                logr = np.log(col.values[1:] / col.values[:-1])
                if len(logr) < self.vol_window:
                    continue
                rv = float(np.std(logr[-self.vol_window:]))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue
                vols[sym] = rv

            if len(vols) < 5:
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                # Sort by vol ascending (lowest vol = best)
                k = min(self.top_k, len(vols))
                ranked = sorted(vols, key=vols.__getitem__)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_weight

        # Build orders
        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY", "JNK"]


NAME = "hy_gated_sp500_lowvol"
HYPOTHESIS = (
    "HY-credit-gated SP500 low-vol factor: hold top-20 SP500 stocks by lowest "
    "63d realized volatility when JNK above 50d SMA AND SPY above 200d SMA; "
    "TLT otherwise. Dual credit+trend gate on a vol-ranked (not momentum-ranked) "
    "SP500 universe. Biweekly rebalance."
)

UNIVERSE = _universe

STRATEGY = HyGatedSP500LowVol()
