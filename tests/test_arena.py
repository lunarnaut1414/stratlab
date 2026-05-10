"""Tests for the stratlab.arena harness: leaderboard, submit, promote.

The end-to-end submit tests run a real backtest against synthetic OHLCV
data; they don't hit yfinance and don't write to the real ``tmp/arena/``
directory. ``arena_paths`` redirects all writes into a per-test
``tmp_path`` and ``fake_data`` patches ``load_universe`` + ``load_bars``
to return ramping synthetic prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stratlab.arena import config
from stratlab.arena import intents as intents_mod
from stratlab.arena import leaderboard as lb
from stratlab.arena import submit as submit_mod
from stratlab.arena.cleanup import remove_strategy
from stratlab.arena.promote import _select_for_promotion


_TOGGLE_MODULE = """
from stratlab.strategies.base import Strategy
from stratlab.engine.broker import Order, OrderSide


class _Toggle(Strategy):
    def __init__(self, period=2, size=10.0):
        super().__init__(period=period, size=size)
        self.period = period
        self.size = size
        self.in_pos = False

    def on_bar(self, ctx):
        if ctx.idx < 1:
            return []
        if self.in_pos and ctx.idx % self.period == 0:
            self.in_pos = False
            return [Order(side=OrderSide.SELL, size=self.size)]
        if (not self.in_pos) and ctx.idx % self.period == 1:
            self.in_pos = True
            return [Order(side=OrderSide.BUY, size=self.size)]
        return []

    def on_start(self):
        self.in_pos = False


NAME = "{name}"
HYPOTHESIS = "alternates long/flat for harness testing"
UNIVERSE = ["SPY"]

STRATEGY = _Toggle(period={period})
"""


_BUY_ONCE_MODULE = """
from stratlab.strategies.base import Strategy
from stratlab.engine.broker import Order, OrderSide


class _BuyOnce(Strategy):
    def __init__(self):
        super().__init__()
        self.fired = False

    def on_bar(self, ctx):
        if self.fired or ctx.idx < 1:
            return []
        self.fired = True
        return [Order(side=OrderSide.BUY, size=10.0)]

    def on_start(self):
        self.fired = False


NAME = "buy_once"
HYPOTHESIS = "Buys once and holds — should fail the trade-count gate."
UNIVERSE = ["SPY"]

STRATEGY = _BuyOnce()
"""


@pytest.fixture
def arena_paths(tmp_path, monkeypatch):
    """Redirect arena paths into tmp_path for isolated test runs."""
    arena_dir = tmp_path / "arena"
    monkeypatch.setattr(config, "ARENA_DIR", arena_dir)
    monkeypatch.setattr(config, "LEADERBOARD", arena_dir / "leaderboard.csv")
    monkeypatch.setattr(config, "RETURNS_MATRIX", arena_dir / "returns_matrix.parquet")
    monkeypatch.setattr(config, "TEARSHEETS_DIR", arena_dir / "tearsheets")
    monkeypatch.setattr(config, "DEAD_ENDS", arena_dir / "dead_ends.md")
    monkeypatch.setattr(config, "STRATEGIES_DIR", tmp_path / "strategies")
    yield arena_dir


@pytest.fixture
def fake_data(monkeypatch):
    """Patch load_universe + load_bars to return ramping synthetic OHLCV."""
    def _fake_load_universe(tickers, start, end, **kwargs):
        bars = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
        prices = np.linspace(100.0, 200.0, len(bars))
        df = pd.DataFrame(
            {
                "open": prices, "high": prices, "low": prices, "close": prices,
                "volume": np.full(len(bars), 1e6),
            },
            index=bars,
        )
        return {t: df.copy() for t in tickers}

    monkeypatch.setattr(
        "stratlab.data.universe.load_universe", _fake_load_universe
    )

    def _fake_load_bars(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(
        "stratlab.data.provider.load_bars", _fake_load_bars
    )


def _write_strategy(directory, filename, body):
    path = directory / filename
    path.write_text(body)
    return path


# --- leaderboard ---------------------------------------------------------

def test_read_leaderboard_missing_file_returns_schema_frame(arena_paths):
    df = lb.read_leaderboard()
    assert df.empty
    assert list(df.columns) == lb.COLUMNS


def test_append_row_round_trip(arena_paths):
    lb.append_row({"strategy_id": "alpha", "generation": 0, "is_calmar": 1.5})
    df = lb.read_leaderboard()
    assert len(df) == 1
    assert df.iloc[0]["strategy_id"] == "alpha"
    assert df.iloc[0]["is_calmar"] == 1.5


def test_update_oos_writes_oos_columns_and_stamps_time(arena_paths):
    lb.append_row({"strategy_id": "alpha", "is_calmar": 1.0})
    lb.update_oos("alpha", {"oos_calmar": 0.7, "oos_sharpe": 0.4})
    df = lb.read_leaderboard()
    assert df.iloc[0]["oos_calmar"] == 0.7
    assert df.iloc[0]["oos_sharpe"] == 0.4
    assert pd.notna(df.iloc[0]["oos_evaluated_at"])


def test_update_oos_unknown_strategy_id_raises(arena_paths):
    with pytest.raises(ValueError, match="not in leaderboard"):
        lb.update_oos("nonexistent", {"oos_calmar": 0.5})


def test_top_k_by_filters_min_trades(arena_paths):
    lb.append_row({"strategy_id": "good", "is_calmar": 1.5, "is_n_trades": 100})
    lb.append_row({"strategy_id": "lucky", "is_calmar": 2.0, "is_n_trades": 5})
    df = lb.read_leaderboard()
    top = lb.top_k_by(df, metric="is_calmar", k=2, require_n_trades=50)
    assert list(top["strategy_id"]) == ["good"]


def test_max_corr_to_no_targets_returns_zero():
    new_returns = pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    corr, name = lb.max_corr_to(new_returns, pd.DataFrame(), target_ids=["any"])
    assert corr == 0.0
    assert name == ""


def test_max_corr_to_finds_near_clone():
    idx = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.RandomState(42)
    base = rng.normal(0.0, 0.01, 100)
    existing = pd.DataFrame(
        {
            "near_twin": base + rng.normal(0.0, 0.0005, 100),
            "unrelated": rng.normal(0.0, 0.01, 100),
        },
        index=idx,
    )
    new = pd.Series(base, index=idx)
    corr, name = lb.max_corr_to(new, existing, target_ids=["near_twin", "unrelated"])
    assert name == "near_twin"
    assert abs(corr) > 0.95


def test_loss_mode_corr_isolates_stress_day_relationship():
    """Two return streams that are uncorrelated overall but move together on
    the worst SPY days should show low daily corr but high loss-mode corr —
    the metric's reason for existing."""
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.RandomState(0)
    spy = pd.Series(rng.normal(0.0005, 0.01, 400), index=idx)
    stress = idx[spy <= spy.quantile(0.10)]

    a = pd.Series(rng.normal(0.0, 0.01, 400), index=idx)
    b = pd.Series(rng.normal(0.0, 0.01, 400), index=idx)
    # On stress days both crash with a shared signal plus modest individual noise
    shock = rng.normal(-0.02, 0.005, len(stress))
    a.loc[stress] = shock + rng.normal(0, 0.0005, len(stress))
    b.loc[stress] = shock + rng.normal(0, 0.0005, len(stress))

    existing = pd.DataFrame({"twin": b}, index=idx)
    overall_corr, _ = lb.max_corr_to(a, existing, target_ids=["twin"])
    stress_corr, twin = lb.loss_mode_corr_to(a, existing, target_ids=["twin"], benchmark_returns=spy)
    assert abs(overall_corr) < 0.5
    assert abs(stress_corr) > 0.9
    assert twin == "twin"


def test_loss_mode_corr_returns_zero_without_benchmark():
    idx = pd.bdate_range("2020-01-01", periods=50)
    a = pd.Series(np.linspace(0.001, 0.01, 50), index=idx)
    existing = pd.DataFrame({"x": a}, index=idx)
    corr, name = lb.loss_mode_corr_to(
        a, existing, target_ids=["x"], benchmark_returns=pd.Series(dtype=float),
    )
    assert corr == 0.0 and name == ""


def test_append_returns_round_trip(arena_paths):
    idx = pd.bdate_range("2020-01-01", periods=10)
    s = pd.Series(np.linspace(0.001, 0.01, 10), index=idx)
    lb.append_returns("alpha", s)
    out = lb.read_returns_matrix()
    assert "alpha" in out.columns
    np.testing.assert_allclose(out["alpha"].values, s.values, rtol=1e-6)


# --- submit helpers ------------------------------------------------------

def test_load_strategy_module_happy_path(tmp_path):
    body = _TOGGLE_MODULE.format(name="toggle_test", period=2)
    path = _write_strategy(tmp_path, "toggle_happy.py", body)
    module = submit_mod.load_strategy_module(path)
    assert module.NAME == "toggle_test"
    assert module.STRATEGY is not None


def test_load_strategy_module_missing_export_raises(tmp_path):
    path = _write_strategy(
        tmp_path, "incomplete.py", "NAME = 'incomplete'\n",
    )
    with pytest.raises(ValueError, match="missing required export"):
        submit_mod.load_strategy_module(path)


def test_resolve_universe_list_passes_through():
    assert submit_mod.resolve_universe(["AAPL", "MSFT"]) == ["AAPL", "MSFT"]


def test_resolve_universe_unknown_string_raises():
    with pytest.raises(ValueError, match="unknown UNIVERSE string"):
        submit_mod.resolve_universe("not-a-real-spec")


def test_resolve_universe_popular_etfs_returns_nonempty_list():
    result = submit_mod.resolve_universe("popular_etfs")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(t, str) for t in result)


def test_resolve_universe_leveraged_etfs_returns_nonempty_list():
    for spec in ("inverse_etfs", "leveraged_etfs", "single_stock_leveraged_etfs"):
        result = submit_mod.resolve_universe(spec)
        assert isinstance(result, list) and len(result) > 0, spec


def test_covers_grants_grace_for_post_holiday_cache_start():
    """Cache that begins a few trading days after the requested start (e.g.
    2010-01-04 vs IS_START 2010-01-01, where Jan 1-3 are non-trading) must
    still be treated as covering the window — otherwise every IS backtest
    triggers needless yfinance refreshes for fully-cached tickers."""
    from stratlab.data.provider import _covers

    idx = pd.bdate_range("2010-01-04", "2018-12-31")
    cached = pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)

    assert _covers(cached, pd.Timestamp("2010-01-01"), pd.Timestamp("2018-12-31"))
    assert _covers(cached, pd.Timestamp("2010-01-04"), pd.Timestamp("2018-12-31"))


def test_covers_rejects_cache_starting_too_late():
    """The grace window is bounded — a cache that starts a month after the
    requested window is NOT covered, regardless of how thin the gap is in
    business days."""
    from stratlab.data.provider import _covers

    idx = pd.bdate_range("2010-02-15", "2018-12-31")
    cached = pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)

    assert not _covers(cached, pd.Timestamp("2010-01-01"), pd.Timestamp("2018-12-31"))


def test_covers_still_requires_full_end_coverage():
    from stratlab.data.provider import _covers

    idx = pd.bdate_range("2010-01-04", "2017-06-30")
    cached = pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)

    assert not _covers(cached, pd.Timestamp("2010-01-01"), pd.Timestamp("2018-12-31"))


def test_slug_normalizes_naming():
    assert submit_mod._slug("My Strategy V1") == "my_strategy_v1"
    assert submit_mod._slug("foo--bar  baz") == "foo_bar_baz"


# --- submit end-to-end ---------------------------------------------------

def test_submit_accepts_passing_strategy(tmp_path, arena_paths, fake_data, monkeypatch):
    # Toggle on the synthetic ramp has low absolute Calmar (~0.11). Disable the
    # Calmar gate for this happy-path test — we just want to exercise the
    # accept flow with a strategy that satisfies the trade-count + corr gates.
    monkeypatch.setattr(config, "MIN_CALMAR_IS", 0.0)
    body = _TOGGLE_MODULE.format(name="toggle_pass", period=2)
    path = _write_strategy(tmp_path, "toggle_pass.py", body)
    sid = submit_mod.submit(path, gen=1, agent_id="pytest")
    assert sid == "gen1_toggle_pass"
    df = lb.read_leaderboard()
    assert len(df) == 1
    assert df.iloc[0]["strategy_id"] == sid
    assert df.iloc[0]["is_n_trades"] >= config.MIN_TRADES_IS


def test_submit_rejects_low_calmar(tmp_path, arena_paths, fake_data, monkeypatch):
    # Set the Calmar floor higher than what the toggle produces — should reject.
    monkeypatch.setattr(config, "MIN_CALMAR_IS", 5.0)
    body = _TOGGLE_MODULE.format(name="toggle_lowcalmar", period=2)
    path = _write_strategy(tmp_path, "toggle_lowcalmar.py", body)
    with pytest.raises(SystemExit) as exc:
        submit_mod.submit(path, gen=1, agent_id="pytest")
    assert exc.value.code == 5
    assert lb.read_leaderboard().empty


def test_submit_rejects_low_n_trades(tmp_path, arena_paths, fake_data):
    path = _write_strategy(tmp_path, "buy_once.py", _BUY_ONCE_MODULE)
    with pytest.raises(SystemExit) as exc:
        submit_mod.submit(path, gen=1, agent_id="pytest")
    assert exc.value.code == 2
    assert lb.read_leaderboard().empty


def test_submit_rejects_high_correlation_duplicate(tmp_path, arena_paths, fake_data, monkeypatch):
    monkeypatch.setattr(config, "MIN_CALMAR_IS", 0.0)
    body_first = _TOGGLE_MODULE.format(name="toggle_first", period=2)
    body_clone = _TOGGLE_MODULE.format(name="toggle_clone", period=2)
    p1 = _write_strategy(tmp_path, "toggle_first.py", body_first)
    p2 = _write_strategy(tmp_path, "toggle_clone.py", body_clone)

    submit_mod.submit(p1, gen=1, agent_id="pytest")
    with pytest.raises(SystemExit) as exc:
        submit_mod.submit(p2, gen=1, agent_id="pytest")
    assert exc.value.code == 3
    assert len(lb.read_leaderboard()) == 1


# --- promote -------------------------------------------------------------

def test_select_for_promotion_empty_leaderboard(arena_paths):
    assert _select_for_promotion(top_k=10, strategy_id=None) == []


def test_select_for_promotion_skips_already_evaluated(arena_paths):
    lb.append_row({
        "strategy_id": "evaluated",
        "is_calmar": 2.0,
        "is_n_trades": 100,
        "oos_evaluated_at": "2024-01-01T00:00:00",
    })
    lb.append_row({
        "strategy_id": "pending",
        "is_calmar": 1.0,
        "is_n_trades": 100,
    })
    selected = _select_for_promotion(top_k=10, strategy_id=None)
    assert len(selected) == 1
    assert selected[0]["strategy_id"] == "pending"


def test_select_for_promotion_respects_min_trades(arena_paths):
    lb.append_row({
        "strategy_id": "lucky",
        "is_calmar": 5.0,
        "is_n_trades": 5,  # below gate
    })
    lb.append_row({
        "strategy_id": "honest",
        "is_calmar": 1.0,
        "is_n_trades": 100,
    })
    selected = _select_for_promotion(top_k=10, strategy_id=None)
    assert [s["strategy_id"] for s in selected] == ["honest"]


# --- cleanup -------------------------------------------------------------

def test_cleanup_removes_leaderboard_row_and_returns(arena_paths):
    lb.append_row({"strategy_id": "junk", "is_calmar": 0.1, "is_n_trades": 100})
    lb.append_row({"strategy_id": "good", "is_calmar": 1.0, "is_n_trades": 100})
    idx = pd.bdate_range("2020-01-01", periods=10)
    lb.append_returns("junk", pd.Series(np.linspace(0.001, 0.01, 10), index=idx))
    lb.append_returns("good", pd.Series(np.linspace(-0.001, 0.005, 10), index=idx))

    summary = remove_strategy("junk")

    df = lb.read_leaderboard()
    assert "junk" not in df["strategy_id"].tolist()
    assert "good" in df["strategy_id"].tolist()
    rm = lb.read_returns_matrix()
    assert "junk" not in rm.columns
    assert "good" in rm.columns
    assert any("removed" in a for a in summary["actions"])


def test_cleanup_idempotent_on_missing_id(arena_paths):
    summary = remove_strategy("nonexistent")
    assert "not found" in summary["actions"][0]


# --- intents pre-commit --------------------------------------------------

def test_intent_commit_returns_id_and_persists(arena_paths):
    iid = intents_mod.commit_intent(
        agent_id="sonnet-1", generation=1,
        hypothesis="Long top-20 by 60d realized vol",
    )
    assert iid.startswith("ic_")
    rows = intents_mod.read_intents()
    assert len(rows) == 1
    assert rows[0]["intent_id"] == iid
    assert rows[0]["status"] == "committed"
    assert rows[0]["agent_id"] == "sonnet-1"


def test_intent_filters_by_generation_and_status(arena_paths):
    intents_mod.commit_intent(agent_id="a", generation=1, hypothesis="alpha")
    intents_mod.commit_intent(agent_id="b", generation=1, hypothesis="beta")
    iid_c = intents_mod.commit_intent(agent_id="c", generation=2, hypothesis="gamma")
    intents_mod.mark_status(iid_c, "submitted", strategy_id="gen2_gamma")

    gen1 = intents_mod.read_intents(generation=1)
    assert len(gen1) == 2
    submitted = intents_mod.read_intents(status="submitted")
    assert len(submitted) == 1
    assert submitted[0]["strategy_id"] == "gen2_gamma"


def test_intent_mark_unknown_id_raises(arena_paths):
    with pytest.raises(ValueError, match="not found"):
        intents_mod.mark_status("ic_nope", "submitted")


def test_intent_mark_invalid_status_raises(arena_paths):
    iid = intents_mod.commit_intent(agent_id="a", generation=1, hypothesis="x")
    with pytest.raises(ValueError, match="unknown status"):
        intents_mod.mark_status(iid, "limbo")


def test_submit_with_intent_id_auto_marks_submitted(
    tmp_path, arena_paths, fake_data, monkeypatch,
):
    monkeypatch.setattr(config, "MIN_CALMAR_IS", 0.0)
    iid = intents_mod.commit_intent(
        agent_id="sonnet-1", generation=1,
        hypothesis="toggle test",
    )
    body = _TOGGLE_MODULE.format(name="toggle_intent_pass", period=2)
    path = _write_strategy(tmp_path, "toggle_intent_pass.py", body)
    sid = submit_mod.submit(
        path, gen=1, agent_id="sonnet-1", intent_id=iid,
    )
    rows = intents_mod.read_intents()
    assert len(rows) == 1
    assert rows[0]["status"] == "submitted"
    assert rows[0]["strategy_id"] == sid


def test_submit_rejection_marks_intent_abandoned(
    tmp_path, arena_paths, fake_data, monkeypatch,
):
    iid = intents_mod.commit_intent(
        agent_id="sonnet-1", generation=1,
        hypothesis="buy-once test",
    )
    path = _write_strategy(tmp_path, "buy_once_intent.py", _BUY_ONCE_MODULE)
    with pytest.raises(SystemExit) as exc:
        submit_mod.submit(path, gen=1, agent_id="sonnet-1", intent_id=iid)
    assert exc.value.code == 2
    rows = intents_mod.read_intents()
    assert rows[0]["status"] == "abandoned"
    assert "n_trades" in rows[0]["notes"]
