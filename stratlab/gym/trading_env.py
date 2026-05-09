from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from stratlab.engine.broker import Broker, Order, OrderSide


class TradingEnv(gym.Env):
    """Gymnasium-compatible trading environment for RL agents.

    Observation: rolling window of OHLCV features + position info.
    Actions: 0 = hold, 1 = buy, 2 = sell.
    Reward: change in portfolio value (risk-adjusted optional).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 20,
        initial_cash: float = 100_000.0,
        trade_size: float = 100.0,
        reward_mode: str = "pnl",
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.trade_size = trade_size
        self.reward_mode = reward_mode

        n_features = len([c for c in df.columns if c in ["open", "high", "low", "close", "volume"]])
        # observation: window of normalized OHLCV + [position_size, unrealized_pnl]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * n_features + 2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # hold, buy, sell

        self.broker = Broker(initial_cash=initial_cash)
        self.current_step = 0
        self.n_features = n_features

    def _get_obs(self) -> np.ndarray:
        start = max(0, self.current_step - self.window_size + 1)
        end = self.current_step + 1

        cols = [c for c in self.df.columns if c in ["open", "high", "low", "close", "volume"]]
        window = self.df.iloc[start:end][cols].values

        # pad if we don't have enough history
        if len(window) < self.window_size:
            pad = np.zeros((self.window_size - len(window), self.n_features))
            window = np.vstack([pad, window])

        # normalize to percentage changes from first row
        base = window[0].copy()
        base[base == 0] = 1.0
        window = window / base - 1.0

        pos = self.broker.get_position("asset")
        price = float(self.df.iloc[self.current_step]["close"])
        unrealized = pos.size * (price - pos.avg_entry) if pos.size > 0 else 0.0

        features = np.concatenate([
            window.flatten(),
            [pos.size / self.trade_size, unrealized / self.initial_cash],
        ])
        return features.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.broker.reset()
        self.current_step = self.window_size
        return self._get_obs(), {}

    def step(self, action: int):
        price = float(self.df.iloc[self.current_step]["close"])
        prev_value = self.broker.portfolio_value({"asset": price})

        if action == 1:  # buy
            order = Order(side=OrderSide.BUY, size=self.trade_size, symbol="asset")
            self.broker.fill_order(order, self.df.iloc[self.current_step], pd.Timestamp.now())
        elif action == 2:  # sell
            pos = self.broker.get_position("asset")
            if pos.size > 0:
                order = Order(side=OrderSide.SELL, size=pos.size, symbol="asset")
                self.broker.fill_order(order, self.df.iloc[self.current_step], pd.Timestamp.now())

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        new_price = float(self.df.iloc[self.current_step]["close"])
        new_value = self.broker.portfolio_value({"asset": new_price})

        if self.reward_mode == "pnl":
            reward = (new_value - prev_value) / self.initial_cash
        elif self.reward_mode == "log_return":
            reward = float(np.log(new_value / prev_value)) if prev_value > 0 else 0.0
        else:
            reward = (new_value - prev_value) / self.initial_cash

        return self._get_obs(), reward, terminated, truncated, {"portfolio_value": new_value}
