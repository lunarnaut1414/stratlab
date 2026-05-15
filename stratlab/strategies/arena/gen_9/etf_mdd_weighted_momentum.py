"""gen_9 sonnet-3 — Popular ETF Multi-Timeframe Momentum Weighted by Inverse Max-Drawdown

Hypothesis: Rank popular ETFs by a composite of 20d+63d+126d momentum percentile scores,
then weight holdings inversely by their 252d maximum drawdown (smooth-uptrend selection).
JNK 30d SMA credit gate for equity ETF routing vs TLT/IEF defensive.

Rationale:
- Multi-timeframe momentum composite (short + medium + long) provides more robust
  signal than any single lookback
- Weighting by inverse max-drawdown (not inverse vol) selects ETFs with the SMOOTHEST
  recent uptrends — drawdown captures tail risk better than realized vol
- JNK credit gate separates risk-on vs risk-off without stock-picking
- ETF universe = low corr to SP500 individual-stock strategies
- NOT present on leaderboard: prior ETF momentum strategies use single lookback or
  inverse-vol weighting; none use max-drawdown weighting or multi-timeframe composites

Distinction:
- gen8_smh_ief_growth_gate, gen8_igv_software_qqq_timing: single signal routing, not ETF ranking
- gen5_atr_momentum_etf: ATR-scaled (short-term noise), single-timeframe, rejected Calmar
- gen7_vix_percentile_vol_target: VIX percentile only, single-asset SPY allocation
- All prior ETF cross-sectional strategies use single-lookback and equal/vol weighting

Universe: popular_etfs (broad ETFs including SPY, QQQ, IWM, GLD, TLT, XLK, XLV, ...)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Use popular_etfs universe
UNIVERSE = "popular_etfs"

REBALANCE_EVERY = 10    # biweekly
SHORT_WINDOW = 20       # short-term momentum component
MID_WINDOW = 63         # medium-term component
LONG_WINDOW = 126       # long-term component
MDD_WINDOW = 252        # 252d max drawdown lookback for weighting
JNK_MA_WINDOW = 30      # JNK credit gate SMA
TOP_K = 3               # top ETFs to hold
EXPOSURE = 0.97
_JNK = "JNK"
_TLT = "TLT"
_IEF = "IEF"
# ETFs to skip (signal-only or defensive)
SKIP_TICKERS = {"TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "JNK", "GLD",
                "^VIX", "^TNX", "^TYX", "^IRX", "^FVX"}


class EtfMddWeightedMomentum(Strategy):
    """ETF multi-timeframe momentum with inverse max-drawdown weighting and JNK credit gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        short_window: int = SHORT_WINDOW,
        mid_window: int = MID_WINDOW,
        long_window: int = LONG_WINDOW,
        mdd_window: int = MDD_WINDOW,
        jnk_ma_window: int = JNK_MA_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            short_window=short_window,
            mid_window=mid_window,
            long_window=long_window,
            mdd_window=mdd_window,
            jnk_ma_window=jnk_ma_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.short_window = int(short_window)
        self.mid_window = int(mid_window)
        self.long_window = int(long_window)
        self.mdd_window = int(mdd_window)
        self.jnk_ma_window = int(jnk_ma_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.long_window, self.mdd_window, self.jnk_ma_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # --- JNK credit gate ---
        credit_risk_on = True
        try:
            jnk_hist = ctx.history(_JNK)
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma_window + 2:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma_window:
                    jnk_sma = float(jnk_close.iloc[-self.jnk_ma_window:].mean())
                    jnk_now = float(jnk_close.iloc[-1])
                    credit_risk_on = jnk_now > jnk_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not credit_risk_on:
            # Credit weak: defensive TLT+IEF 60/37
            if _TLT in live:
                target[_TLT] = self.exposure * 0.618
            if _IEF in live:
                target[_IEF] = self.exposure * 0.382
        else:
            # Credit healthy: rank ETFs by multi-timeframe composite
            need = max(self.long_window, self.mdd_window) + 10
            prices = ctx.closes_window(need)
            if len(prices) < self.long_window:
                if "SPY" in live:
                    target["SPY"] = self.exposure
            else:
                # Compute returns for each timeframe
                short_rets: dict[str, float] = {}
                mid_rets: dict[str, float] = {}
                long_rets: dict[str, float] = {}
                mdd_vals: dict[str, float] = {}

                eligible_syms = [s for s in prices.columns
                                  if s not in SKIP_TICKERS and s in live]

                for sym in eligible_syms:
                    col = prices[sym].dropna()
                    if len(col) < self.long_window:
                        continue

                    p_now = float(col.iloc[-1])
                    if p_now <= 0:
                        continue

                    # Short momentum (20d)
                    if len(col) >= self.short_window:
                        r_short = p_now / float(col.iloc[-self.short_window]) - 1.0
                        if np.isfinite(r_short):
                            short_rets[sym] = r_short

                    # Mid momentum (63d)
                    if len(col) >= self.mid_window:
                        r_mid = p_now / float(col.iloc[-self.mid_window]) - 1.0
                        if np.isfinite(r_mid):
                            mid_rets[sym] = r_mid

                    # Long momentum (126d)
                    if len(col) >= self.long_window:
                        r_long = p_now / float(col.iloc[-self.long_window]) - 1.0
                        if np.isfinite(r_long):
                            long_rets[sym] = r_long

                    # Max drawdown over 252d (for weighting)
                    if len(col) >= self.mdd_window:
                        window_prices = col.iloc[-self.mdd_window:]
                    else:
                        window_prices = col
                    running_max = window_prices.cummax()
                    drawdowns = (window_prices / running_max) - 1.0
                    mdd = float(drawdowns.min())  # most negative value
                    if np.isfinite(mdd):
                        mdd_vals[sym] = mdd  # negative number

                # Only include symbols with all three momentum windows
                valid = set(short_rets) & set(mid_rets) & set(long_rets) & set(mdd_vals)
                if len(valid) < self.top_k:
                    if "SPY" in live:
                        target["SPY"] = self.exposure
                else:
                    # Compute percentile ranks for each window
                    def pct_rank(d: dict) -> dict[str, float]:
                        vals = sorted(d.items(), key=lambda x: x[1])
                        n = len(vals)
                        return {sym: i / (n - 1) if n > 1 else 0.5
                                for i, (sym, _) in enumerate(vals)}

                    valid_short = {s: short_rets[s] for s in valid}
                    valid_mid = {s: mid_rets[s] for s in valid}
                    valid_long = {s: long_rets[s] for s in valid}

                    pct_s = pct_rank(valid_short)
                    pct_m = pct_rank(valid_mid)
                    pct_l = pct_rank(valid_long)

                    # Composite: equal weight across 3 timeframes
                    composite: dict[str, float] = {}
                    for sym in valid:
                        composite[sym] = (pct_s[sym] + pct_m[sym] + pct_l[sym]) / 3.0

                    # Take top-K by composite score with positive long-term momentum
                    ranked = sorted(composite, key=composite.__getitem__, reverse=True)
                    candidates = [s for s in ranked if long_rets.get(s, -1) > 0][:self.top_k]

                    if not candidates:
                        # Fall back to top-K regardless of absolute momentum
                        candidates = ranked[:self.top_k]

                    if not candidates:
                        if "SPY" in live:
                            target["SPY"] = self.exposure
                    else:
                        # Weight inversely by max-drawdown magnitude (lower MDD = higher weight)
                        inv_mdd_weights: dict[str, float] = {}
                        for sym in candidates:
                            mdd = mdd_vals.get(sym, -0.20)
                            # Convert MDD to weight: smaller MDD (less negative) = more weight
                            # Use 1/(1 + abs(mdd)) so mdd=0 -> 1.0, mdd=-0.5 -> 0.67
                            inv_mdd_weights[sym] = 1.0 / (1.0 + abs(mdd))

                        total_w = sum(inv_mdd_weights.values())
                        for sym, w in inv_mdd_weights.items():
                            if sym in live:
                                target[sym] = self.exposure * w / total_w

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


NAME = "etf_mdd_weighted_momentum"
HYPOTHESIS = (
    "Popular ETF multi-timeframe momentum (20d+63d+126d composite score) weighted by "
    "inverse 252d max-drawdown (smooth-uptrend selection); JNK 30d SMA credit gate for "
    "equity ETF top-3 vs TLT/IEF defensive; rebalance every 10 bars; "
    "drawdown-weighted ETF rotation distinct from vol-weighted and equal-weight prior approaches"
)

STRATEGY = EtfMddWeightedMomentum()
