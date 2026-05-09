"""daily_sentiment aggregation.

Pure data plumbing, but bugs here would silently corrupt every
news-driven signal an agent computes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stratlab.news.features import daily_sentiment


def _write_day(news_dir: Path, source: str, topic: str, day: str,
               articles: dict) -> None:
    path = news_dir / source / topic / day[:4] / f"{source}-{topic}-{day}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(articles))


def _article(net: float, pos: float = 0.0, neg: float = 0.0,
             neutral: float = 0.0, has_sentiment: bool = True) -> dict:
    out = {"title": "x", "content": "x", "published_date": "2024-01-01"}
    if has_sentiment:
        out["sentiment"] = {
            "pos": pos, "neg": neg, "neutral": neutral, "net": net,
            "model": "ProsusAI/finbert",
        }
    return out


def test_daily_sentiment_empty_dir_returns_empty_frame(tmp_path: Path):
    out = daily_sentiment(news_dir=tmp_path)
    assert out.empty


def test_daily_sentiment_aggregates_mean_across_articles(tmp_path: Path):
    _write_day(tmp_path, "npr", "business", "2024-01-15", {
        "a1": _article(net=0.6, pos=0.7, neg=0.1, neutral=0.2),
        "a2": _article(net=0.2, pos=0.3, neg=0.1, neutral=0.6),
    })
    out = daily_sentiment(news_dir=tmp_path)
    # default: net only, MultiIndex columns (source, topic)
    val = out.loc[pd.Timestamp("2024-01-15"), ("npr", "business")]
    assert val == pytest.approx(0.4)  # mean(0.6, 0.2)


def test_daily_sentiment_ignores_unscored_articles(tmp_path: Path):
    """Articles without a sentiment field must be silently skipped."""
    _write_day(tmp_path, "npr", "business", "2024-01-15", {
        "a1": _article(net=0.6),
        "a2": _article(net=0.0, has_sentiment=False),  # unscored
    })
    out = daily_sentiment(news_dir=tmp_path)
    val = out.loc[pd.Timestamp("2024-01-15"), ("npr", "business")]
    assert val == pytest.approx(0.6)  # only a1 counted


def test_daily_sentiment_breakdown_includes_count_and_components(tmp_path: Path):
    _write_day(tmp_path, "npr", "economy", "2024-02-01", {
        "a1": _article(net=0.4, pos=0.5, neg=0.1, neutral=0.4),
        "a2": _article(net=-0.2, pos=0.1, neg=0.3, neutral=0.6),
    })
    out = daily_sentiment(news_dir=tmp_path, breakdown=True)
    row = out.loc[pd.Timestamp("2024-02-01")]
    assert row[("npr", "economy", "article_count")] == 2
    assert row[("npr", "economy", "net")] == pytest.approx(0.1)  # mean(0.4, -0.2)
    assert row[("npr", "economy", "pos")] == pytest.approx(0.3)


def test_daily_sentiment_filters_by_date_range(tmp_path: Path):
    _write_day(tmp_path, "npr", "world", "2024-01-15",
               {"a1": _article(net=0.5)})
    _write_day(tmp_path, "npr", "world", "2024-06-15",
               {"a1": _article(net=-0.5)})
    _write_day(tmp_path, "npr", "world", "2024-12-15",
               {"a1": _article(net=0.0)})

    out = daily_sentiment(start="2024-04-01", end="2024-09-30",
                         news_dir=tmp_path)
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2024-06-15")


def test_daily_sentiment_filters_by_source_and_topic(tmp_path: Path):
    _write_day(tmp_path, "npr", "business", "2024-01-15",
               {"a1": _article(net=0.5)})
    _write_day(tmp_path, "ap", "business", "2024-01-15",
               {"a1": _article(net=-0.5)})
    _write_day(tmp_path, "npr", "world", "2024-01-15",
               {"a1": _article(net=0.0)})

    out = daily_sentiment(sources=["npr"], topics=["business"], news_dir=tmp_path)
    assert list(out.columns) == [("npr", "business")]
