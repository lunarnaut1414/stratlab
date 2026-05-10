# Stratlab Arena — Pilot Run Guide

This is the manual procedure for the **pilot** version of the arena: a
small 3-generation × 4-agent run designed to validate the protocol
before paying for a full 30-generation arena.

## Goals of the pilot

1. Confirm `submit.py` and `promote.py` work end-to-end
2. Confirm the generator prompt produces sensible strategies (not just
   indicator copy-paste)
3. Confirm the corr-rejection filter is calibrated reasonably (not
   rejecting everything, not letting duplicates through)
4. See what the curator memo looks like — does it add signal beyond
   ranking?
5. Estimate per-generation wall-clock + cost so you can budget the full run

If anything in this list breaks, fix it before running 30 generations.

## Prerequisites

- Stratlab data refreshed for IS + OOS windows:
  ```bash
  python -m stratlab.refresh
  ```
  Verify SPY, SH, and S&P 500 constituents have data from 2010-01-01
  through 2024-12-31. The seed strategies need this.
- Tests passing:
  ```bash
  pytest tests/test_arena.py
  ```

## Round 0 — seeds

Bootstrap the leaderboard with the three known-good templates:

```bash
python -m stratlab.arena.seed
```

You should see three "ACCEPTED" lines and a populated
`tmp/arena/leaderboard.csv` (3 rows, gen=0). If any seed fails, the
harness or data is broken — debug before proceeding.

**Sanity-check the seeds before continuing.** Open
`tmp/arena/tearsheets/gen0_*.html` in a browser. Equity curves should
look like real strategies, not flat lines or moonshots. Calmar should
be in a reasonable range (0.3–1.5 ish for these particular seeds over
2010-2018).

## Rounds 1–3 — generation

For each generation `N` in `{1, 2, 3}`:

### Generators (4 in parallel)

Open four Claude Code windows. Paste the generator prompt
(`docs/arena/generator_prompt.md`) into each, with this header:

```
ROUND CONTEXT
- Generation: <N>
- Your agent_id: claude-sonnet-<i>     (i = 1, 2, 3, 4)
- Submit with: python -m stratlab.arena.submit \
    stratlab/strategies/arena/gen_<N>/<your_slug>.py \
    --gen <N> --agent-id sonnet-<i>
```

Each window writes one strategy and submits it. Expected outcomes per
round:

- 4 submissions attempted
- 0–2 rejections (trade-count gate or corr filter)
- 2–4 accepted onto the leaderboard

If 4/4 are rejected, the prompt or seeds need adjustment. If 4/4 are
accepted but they're all near-clones of each other (visually similar
equity curves), the corr filter is mis-calibrated for the round-1
small-population regime — note for follow-up.

### Read what was produced

Before kicking off the next generation, **manually inspect** every
accepted strategy:

- Source code: does the hypothesis match the implementation?
- Tearsheet: does the equity curve look like real trading or a
  curve-fit?
- IS metrics: is Calmar coming from genuine compounding or one big bet?

If you spot a look-ahead leak (e.g., agent used `ctx._closes_df` instead
of `ctx.closes()`), kill the strategy file and remove the row from the
leaderboard manually. Note the failure pattern in `dead_ends.md` so the
generator prompt can be tightened.

## Curator round (after round 3)

```bash
# 1. Run OOS sweep on the top entries
python -m stratlab.arena.promote --top 10

# 2. Open Claude Code with Opus and paste the curator prompt
#    (docs/arena/curator_prompt.md)
#
#    Round context:
#    - Round number: 1 (first curator round)
#    - Strategies submitted this round: ~9-12 (3 gens × 4 agents minus rejections)
#    - Cumulative population: same (pilot only ran 3 gens)
#    - Promotion budget: top 5
```

Curator produces four artifacts under `tmp/arena/round_1/`. Read the
memo. Sanity-check:

- Does it identify legitimate failure modes, or is it just summarizing
  the leaderboard?
- Are the ensemble proposals genuinely uncorrelated?
- Does the OOS performance match what you'd predict from the IS
  metrics? (Some IS-OOS gap is expected; collapse is suspicious.)

## Pilot exit criteria

Before scaling to 30 generations, confirm:

- [ ] All seeds were submitted and OOS-evaluated successfully
- [ ] At least 50% of generator submissions were accepted (rest rejected
      for legitimate reasons, not infrastructure bugs)
- [ ] No strategy in the leaderboard has a look-ahead leak
- [ ] At least one accepted strategy has IS Calmar > 0.7 AND OOS
      Calmar > 0.4 (i.e., something works on both windows)
- [ ] Curator memo identifies at least one non-trivial observation
      (e.g., "the population is monocultured around momentum" or
      "strategy X has IS-OOS collapse pattern that looks like
      regime-specific overfit")
- [ ] Total spend across the pilot is < $30 (Claude Max plan absorbs
      this; tracking is just to estimate full-run cost)

If all six pass, scale up. If any fail, diagnose before scaling.

## Common failure modes

- **All submissions rejected by corr filter.** Top-5 is dominated by one
  family of strategies. Either reduce the threshold for round 1
  (population is too small to enforce 0.85 strictly) or seed with a
  more diverse gen-0.
- **Submissions accept but n_trades is borderline.** Generators are
  finding the gate and tuning to it. Tighten the gate (raise
  MIN_TRADES_IS) so the optimum sits clearly above the floor.
- **IS Calmar excellent, OOS Calmar collapses.** Generators are
  overfitting in-sample. Either narrow the hyperparameter space the
  prompt encourages, or add a regularization penalty (e.g., reject
  parameters tuned to the bar).
- **All strategies have similar equity-curve shape despite passing the
  corr filter.** The 0.85 threshold may be too lax — equity curves with
  daily-return corr 0.6 can still look visually clone-y.
