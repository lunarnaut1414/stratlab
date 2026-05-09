"""Curated technical-analysis primitives.

Thin facade over `ta <https://github.com/bukosabino/ta>`_ — no
reimplementation. Exposes a stable ``stratlab.indicators`` namespace so
strategies can ``from stratlab.indicators import rsi, macd, atr`` without
caring which sub-module of ``ta`` an indicator happens to live in.

All functions are pure: they take pandas ``Series`` (or ``high/low/close``
/``volume`` for multi-input indicators) and return a ``Series`` of the
same index. They are safe to call inside ``Strategy.on_bar`` on the
sliced ``ctx.history()`` frame — no look-ahead.

The 25 primitives below cover the standard quant toolbelt. If you need
something more exotic (Williams %R, Ulcer Index, Vortex, KAMA, etc.),
import directly from ``ta.momentum`` / ``ta.volatility`` / etc — they're
all functional and follow the same shape.
"""
from __future__ import annotations

from ta.momentum import (
    rsi,
    roc,
    stoch,
    stoch_signal,
)
from ta.trend import (
    adx,
    aroon_down,
    aroon_up,
    cci,
    ema_indicator as ema,
    macd,
    macd_diff,
    macd_signal,
    sma_indicator as sma,
    wma_indicator as wma,
)
from ta.volatility import (
    average_true_range as atr,
    bollinger_hband as bb_upper,
    bollinger_lband as bb_lower,
    bollinger_mavg as bb_middle,
    bollinger_pband as bb_pband,
    donchian_channel_hband as donchian_upper,
    donchian_channel_lband as donchian_lower,
)
from ta.volume import (
    chaikin_money_flow as cmf,
    money_flow_index as mfi,
    on_balance_volume as obv,
    volume_weighted_average_price as vwap,
)

__all__ = [
    # Trend
    "sma", "ema", "wma",
    "macd", "macd_signal", "macd_diff",
    "adx", "aroon_up", "aroon_down", "cci",
    # Momentum
    "rsi", "roc", "stoch", "stoch_signal",
    # Volatility
    "atr",
    "bb_upper", "bb_lower", "bb_middle", "bb_pband",
    "donchian_upper", "donchian_lower",
    # Volume
    "obv", "mfi", "cmf", "vwap",
]
