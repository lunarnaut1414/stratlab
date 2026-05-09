"""News scrapers — one submodule per source.

Each source module exposes a ``scrape(...)`` function that writes per-article
records into ``data/news/<source>/<category>/<year>.json``. Articles are keyed
by a stable ID so re-running is idempotent (already-saved articles are skipped).
"""
