"""JNK/LQD Credit Z-Score with Idiosyncratic Momentum + QQQ Neutral — gen_9 sonnet-8

Hypothesis:
Use JNK/LQD 90d rolling z-score credit regime gate with:
  - z > +0.5 (credit tightening): hold top-15 SP500 stocks by idiosyncratic
    momentum (63d beta-adjusted alpha: return - beta * SPY return)
  - -0.5 <= z <= +0.5 (neutral credit): hold QQQ 97% (tech alpha in neutral regime)
  - z < -0.5 (credit widening): hold TLT 97% (full defensive)
  - SPY 200d SMA outer bear gate overrides all -> TLT 97%
Rebalance every 10 bars (biweekly).

Rationale:
This combines:
1. The gen_8 best credit signal (JNK/LQD 90d z-score, IS Calmar 0.88)
2. The gen_7 best stock selection (idiosyncratic momentum, IS Calmar 1.20, OOS 0.70)
3. Novel neutral-tier: QQQ instead of IEF+SPY blend

Key distinction from gen8_sp500_credit_zscore_3tier:
- Risk-on branch uses IDIOSYNCRATIC momentum (beta-adjusted) not raw momentum
  -> selects stocks with genuine alpha, not just market beta
- Neutral tier: QQQ 97% not IEF+SPY blend
  -> captures tech alpha in moderate-credit regimes
  -> different daily return path and lower corr to SP500 xsect strategies

The idiosyncratic momentum selection filters out high-beta cyclicals that often
appear in raw-momentum rankings, reducing loss-mode correlation vs SPY-heavy
momentum strategies.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

MOMENTUM_WINDOW = 63      # 3-month window for idiosyncratic momentum
BETA_WINDOW = 126         # 6-month window for beta estimation
TREND_WINDOW = 200        # SPY 200d SMA bear gate
ZSCORE_WINDOW = 90        # Rolling window for JNK/LQD z-score
Z_HIGH = 0.5              # Above this: risk-on (idiosyncratic stock selection)
Z_LOW = -0.5              # Below this: risk-off (TLT)
TOP_K = 15
REBALANCE_DAYS = 10       # Biweekly
EXPOSURE = 0.97


def _universe() -> list[str]:
    return sp500_tickers() + ["JNK", "LQD", "TLT", "QQQ", "SPY"]


UNIVERSE = _universe


class CreditZscoreIdioMomQQQ(Strategy):
    """JNK/LQD z-score gating idiosyncratic SP500 momentum / QQQ / TLT."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, ZSCORE_WINDOW, MOMENTUM_WINDOW, BETA_WINDOW) + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live_all = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # --- SPY 200d SMA bear gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_bear = False
        if len(spy_hist) >= TREND_WINDOW:
            spy_close = spy_hist["close"].dropna()
            spy_sma = float(spy_close.iloc[-TREND_WINDOW:].mean())
            spy_price = live_all.get("SPY", 0.0)
            spy_bear = spy_price > 0 and spy_price <= spy_sma

        if spy_bear:
            target: dict[str, float] = {"TLT": EXPOSURE}
        else:
            # --- Compute JNK/LQD ratio z-score ---
            try:
                jnk_hist = ctx.history("JNK")
                lqd_hist = ctx.history("LQD")
            except KeyError:
                return []

            if len(jnk_hist) < ZSCORE_WINDOW + 5 or len(lqd_hist) < ZSCORE_WINDOW + 5:
                return []

            jnk_close = jnk_hist["close"].tail(ZSCORE_WINDOW + 5)
            lqd_close = lqd_hist["close"].tail(ZSCORE_WINDOW + 5)
            min_len = min(len(jnk_close), len(lqd_close))
            jnk_vals = jnk_close.values[-min_len:]
            lqd_vals = lqd_close.values[-min_len:]
            lqd_safe = np.where(lqd_vals > 0, lqd_vals, np.nan)
            ratio = jnk_vals / lqd_safe
            ratio_window = ratio[-ZSCORE_WINDOW:]
            valid = ratio_window[~np.isnan(ratio_window)]
            if len(valid) < 20:
                return []
            ratio_mean = float(np.mean(valid))
            ratio_std = float(np.std(valid))
            if ratio_std <= 0 or not np.isfinite(ratio_std):
                return []
            current_ratio = valid[-1]
            z_score = (current_ratio - ratio_mean) / ratio_std

            if z_score < Z_LOW:
                # Credit widening: TLT
                target = {"TLT": EXPOSURE}
            elif z_score <= Z_HIGH:
                # Neutral credit: QQQ
                target = {"QQQ": EXPOSURE}
            else:
                # Credit tightening: idiosyncratic SP500 momentum
                need = max(MOMENTUM_WINDOW, BETA_WINDOW) + 5
                prices_window = ctx.closes_window(need)
                if len(prices_window) < MOMENTUM_WINDOW:
                    target = {"QQQ": EXPOSURE}
                else:
                    # Get SPY returns for beta estimation
                    spy_close_series = spy_hist["close"].dropna()
                    if len(spy_close_series) < BETA_WINDOW + 1:
                        target = {"QQQ": EXPOSURE}
                    else:
                        spy_tail = spy_close_series.iloc[-(BETA_WINDOW + 1):]
                        spy_logrets = np.log(spy_tail.values[1:] / spy_tail.values[:-1])
                        spy_var = float(np.var(spy_logrets))

                        live = {s: float(closes[s]) for s in closes.index
                                if closes[s] > 0 and s not in ("JNK", "LQD", "TLT", "QQQ", "SPY")}

                        idio_scores: dict[str, float] = {}
                        for sym in live:
                            if sym not in prices_window.columns:
                                continue
                            col = prices_window[sym].dropna()
                            if len(col) < max(MOMENTUM_WINDOW, BETA_WINDOW) + 1:
                                continue

                            # Compute idiosyncratic momentum over MOMENTUM_WINDOW
                            p_end = float(col.iloc[-1])
                            p_start = float(col.iloc[-MOMENTUM_WINDOW])
                            if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                                continue
                            raw_ret = p_end / p_start - 1.0

                            # Compute beta via log returns over BETA_WINDOW
                            tail_beta = col.iloc[-(BETA_WINDOW + 1):]
                            if len(tail_beta) < BETA_WINDOW + 1:
                                continue
                            stock_logrets = np.log(tail_beta.values[1:] / tail_beta.values[:-1])
                            if spy_var <= 1e-10:
                                beta = 1.0
                            else:
                                beta = float(np.cov(stock_logrets, spy_logrets)[0, 1]) / spy_var

                            # SPY total return over MOMENTUM_WINDOW
                            spy_mom = float(spy_close_series.iloc[-1]) / float(spy_close_series.iloc[-MOMENTUM_WINDOW]) - 1.0

                            # Idiosyncratic alpha = raw return - beta * SPY return
                            idio = raw_ret - beta * spy_mom

                            if np.isfinite(idio):
                                idio_scores[sym] = idio

                        if len(idio_scores) < TOP_K:
                            target = {"QQQ": EXPOSURE}
                        else:
                            ranked = sorted(idio_scores, key=idio_scores.__getitem__, reverse=True)[:TOP_K]
                            # Apply 200d SMA trend filter on selected stocks
                            selected = []
                            for sym in ranked:
                                if len(selected) >= TOP_K:
                                    break
                                try:
                                    hist = ctx.history(sym)
                                except KeyError:
                                    continue
                                if len(hist) < TREND_WINDOW:
                                    continue
                                sma = float(hist["close"].iloc[-TREND_WINDOW:].mean())
                                price = live.get(sym, 0.0)
                                if price > sma:
                                    selected.append(sym)

                            if not selected:
                                target = {"QQQ": EXPOSURE}
                            else:
                                target = {sym: EXPOSURE / len(selected) for sym in selected}

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live_all.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        if equity <= 0:
            return []

        # Build orders
        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in target.items():
            price = live_all.get(sym)
            if price is None or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "gen9_credit_zscore_idio_mom_qqq"
HYPOTHESIS = (
    "JNK/LQD 90d z-score: z>+0.5 hold top-15 SP500 by idiosyncratic momentum "
    "(63d beta-adjusted alpha); -0.5 to +0.5 hold QQQ 97%; z<-0.5 hold TLT 97%; "
    "SPY 200d bear gate; biweekly rebalance."
)

STRATEGY = CreditZscoreIdioMomQQQ()
