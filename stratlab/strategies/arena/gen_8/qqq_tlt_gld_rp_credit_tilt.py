"""QQQ/TLT/GLD Risk Parity with JNK Credit Tilt — gen_8 sonnet-2

Hypothesis: Hold QQQ, TLT, and GLD in inverse-volatility (risk parity) weights,
always invested. When JNK is above its 20d SMA (credit risk-on), tilt the QQQ
weight upward by 25% of its RP share. When JNK is below (credit stress), tilt
down by 15% toward TLT. Biweekly rebalance.

Rationale:
- Risk parity with QQQ (not SPY) captures tech growth premium within parity framework
- GLD as third asset provides inflation/safe-haven diversification absent from SPY+TLT
- JNK credit tilt adjusts between growth (QQQ) and safety (TLT) within the parity base
- Always-invested reduces timing risk vs binary regime switchers
- Distinct from gen6_rp_credit_tilt (uses SPY not QQQ; only 2 assets; different tilt)
- QQQ instead of SPY gives higher growth beta in the equity bucket

The always-invested parity structure creates low correlation with pure equity momentum
strategies because the GLD+TLT ballast smooths out equity drawdowns.

Rebalance: every 10 bars (biweekly)
Vol estimation: 20d realized vol for each of QQQ, TLT, GLD
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
VOL_WINDOW = 20           # 20d for realized vol estimation
EXPOSURE = 0.97
JNK_MA_WIN = 20           # JNK 20d SMA
CREDIT_RISK_ON_TILT = 0.25    # when JNK strong, add 25% of QQQ RP weight to QQQ
CREDIT_RISK_OFF_TILT = 0.15   # when JNK weak, shift 15% of QQQ RP weight to TLT
_QQQ = "QQQ"
_TLT = "TLT"
_GLD = "GLD"
_JNK = "JNK"


class QQQTLTGLDRPCreditTilt(Strategy):
    """QQQ/TLT/GLD risk parity with JNK credit tilt."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vol_window: int = VOL_WINDOW,
        exposure: float = EXPOSURE,
        jnk_ma_win: int = JNK_MA_WIN,
        credit_risk_on_tilt: float = CREDIT_RISK_ON_TILT,
        credit_risk_off_tilt: float = CREDIT_RISK_OFF_TILT,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vol_window=vol_window,
            exposure=exposure,
            jnk_ma_win=jnk_ma_win,
            credit_risk_on_tilt=credit_risk_on_tilt,
            credit_risk_off_tilt=credit_risk_off_tilt,
        )
        self.rebalance_every = int(rebalance_every)
        self.vol_window = int(vol_window)
        self.exposure = float(exposure)
        self.jnk_ma_win = int(jnk_ma_win)
        self.credit_risk_on_tilt = float(credit_risk_on_tilt)
        self.credit_risk_off_tilt = float(credit_risk_off_tilt)

    def _inv_vol_weight(self, sym: str, ctx: BarContext) -> float | None:
        """Compute inverse-vol weight denominator (just 1/vol)."""
        try:
            hist = ctx.history(sym)
        except KeyError:
            return None
        close = hist["close"].dropna()
        if len(close) < self.vol_window + 2:
            return None
        tail = close.iloc[-(self.vol_window + 1):]
        log_rets = np.log(tail.values[1:] / tail.values[:-1])
        vol = float(np.std(log_rets))
        if vol <= 1e-8 or not np.isfinite(vol):
            return None
        return 1.0 / vol

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.vol_window + self.jnk_ma_win + 5
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

        # Compute inverse-vol for each asset
        inv_vols: dict[str, float] = {}
        for sym in [_QQQ, _TLT, _GLD]:
            iv = self._inv_vol_weight(sym, ctx)
            if iv is not None:
                inv_vols[sym] = iv

        if len(inv_vols) < 2:
            return []

        # Risk parity weights (normalized)
        total_iv = sum(inv_vols.values())
        rp_weights = {s: v / total_iv for s, v in inv_vols.items()}

        # JNK credit tilt
        jnk_risk_on = False
        try:
            jnk_hist = ctx.history(_JNK)
            jnk_close = jnk_hist["close"].dropna()
            if len(jnk_close) >= self.jnk_ma_win:
                jnk_ma = float(jnk_close.iloc[-self.jnk_ma_win:].mean())
                jnk_now = float(jnk_close.iloc[-1])
                jnk_risk_on = jnk_now > jnk_ma
        except Exception:
            pass

        # Apply credit tilt: shift weight between QQQ and TLT
        weights = dict(rp_weights)
        if _QQQ in weights and _TLT in weights:
            if jnk_risk_on:
                # Credit risk-on: boost QQQ, reduce TLT
                tilt = self.credit_risk_on_tilt * weights.get(_QQQ, 0.0)
                available = min(tilt, weights.get(_TLT, 0.0))
                weights[_QQQ] = weights.get(_QQQ, 0.0) + available
                weights[_TLT] = weights.get(_TLT, 0.0) - available
            else:
                # Credit risk-off: reduce QQQ, boost TLT
                tilt = self.credit_risk_off_tilt * weights.get(_QQQ, 0.0)
                weights[_QQQ] = weights.get(_QQQ, 0.0) - tilt
                weights[_TLT] = weights.get(_TLT, 0.0) + tilt

        # Normalize and apply total exposure cap
        total_w = sum(weights.values())
        if total_w <= 0:
            return []
        final_weights = {s: (w / total_w) * self.exposure for s, w in weights.items()}

        target = {s: w for s, w in final_weights.items() if s in live and w > 0}

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
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


UNIVERSE = [_QQQ, _TLT, _GLD, _JNK]

NAME = "qqq_tlt_gld_rp_credit_tilt"
HYPOTHESIS = (
    "QQQ/TLT/GLD risk parity with JNK credit tilt: inverse-vol weighted QQQ+TLT+GLD "
    "always-invested; when JNK > 20d SMA (risk-on) tilt 25% of QQQ RP share extra into QQQ; "
    "when JNK weak tilt 15% of QQQ RP into TLT; biweekly rebalance; "
    "distinct from gen6_rp_credit_tilt (QQQ not SPY; adds GLD as 3rd asset)"
)

STRATEGY = QQQTLTGLDRPCreditTilt()
