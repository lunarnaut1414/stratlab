"""Demo: use the TradingEnv with a random agent to verify the gym interface works."""

from stratlab import load_bars
from stratlab.gym.trading_env import TradingEnv


def main():
    print("Fetching SPY daily bars...")
    data = load_bars("SPY", start="2022-01-01", end="2024-01-01")
    print(f"Loaded {len(data)} bars\n")

    env = TradingEnv(df=data, window_size=20, trade_size=50)
    obs, info = env.reset()

    total_reward = 0.0
    steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(f"Random agent results:")
    print(f"  Steps:            {steps}")
    print(f"  Total reward:     {total_reward:.4f}")
    print(f"  Final portfolio:  ${info['portfolio_value']:,.2f}")
    print(f"  Observation shape: {obs.shape}")
    print(f"\nGym env is working. Plug in your RL agent to train!")


if __name__ == "__main__":
    main()
