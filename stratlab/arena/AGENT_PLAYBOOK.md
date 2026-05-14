# Strategies Arena — Agent Orchestration Playbook

This file is a runbook for **Claude Code** (the orchestrator), not for human readers. When the user invokes it (`@stratlab/arena/AGENT_PLAYBOOK.md run`), execute the steps below in order. The harness in `stratlab/arena/` enforces all rules — this playbook only specifies who is spawned, with what brief, and how the orchestrator stitches phases together.

---

## Parameters (defaults; user may override per invocation)

| Param | Default | Notes |
|---|---|---|
| `gen_N` | next unused `gen_<int>/` under `stratlab/strategies/arena/` | e.g. `gen_4` |
| `phase1_generators` | 10 | Sonnet subagents, parallel |
| `phase1_ideas_per_agent` | 3 | each generates + implements + submits |
| `phase2_refiners` | 5 | Opus subagents, fixed roles below |
| `wipe_leaderboard` | false | if true, rotate `tmp/arena/` to `tmp/arena_archive/<ts>/` first |
| `universe_override` | none | otherwise current `default_universe()` |
| `oos_promote_top` | 10 | `promote.py --top` after Phase 2 |

---

## Tool preference for file inspection (use built-ins, not Bash)

The narrow `.claude/settings.json` allowlist only whitelists specific stratlab harness CLI commands. **Ad-hoc Bash one-liners for data exploration trigger permission prompts** that the orchestrator can't always approve cleanly when many subagents run in parallel. Use Claude Code's built-in tools instead — they're always allowed and never prompt:

| To do this | Use this (✅) | NOT this (❌) |
|---|---|---|
| Read a CSV / Python file / config | `Read` tool | `cat`, `head`, `tail`, `python -c "open(..)"` |
| Find files by pattern | `Glob` tool | `find`, `ls **/*.csv` |
| Search for a symbol/string | `Grep` tool | `grep -r`, `rg` |
| Inspect cached coverage of a curated universe | `python -m stratlab.data.inception --universe <name>` | ad-hoc `python -c "import pandas; ..."` snippets |
| List ETF/stock categories and counts | `python -m stratlab.data.catalog show` | walking `data/market/` with `find` |
| Check whether a ticker is cached | `Glob` for `data/market/**/<TICKER>_*.csv` | `ls data/market/*/${TICKER}*` |

### Allowed Bash forms (in `.claude/settings.json`'s allowlist)

- `python -m stratlab.arena.{submit,intents,promote,cleanup,dump_trades,dump_equity_curve,dump_annual_calmar,regime_check,corr_dump,corr_check,is_calmar_estimate,seed} *`
- `python -m stratlab.data.{inception,catalog} *`
- `python -m pytest *` and `./.venv/bin/pytest tests/ *`

### Cheap pre-checks BEFORE wasting a full submit

The arena harness now has several "dry-run" CLIs that cost less than a full
submit and do NOT consume an intent / write to leaderboard / append to
dead_ends. Use them aggressively to iterate on candidates before committing
an intent:

- **`python -m stratlab.arena.regime_check --signal "<expr>"`** — pre-validate
  how often a gate fires in IS. Saves wasted submissions on too-restrictive
  gates. Examples: `"VIX<20"`, `"TLT_21d > IEF_21d"`, `"JNK > JNK_30d_MA"`.
- **`python -m stratlab.arena.is_calmar_estimate <strategy_path>`** — run a
  sub-window backtest (2010-2014 by default) to estimate IS Calmar before
  committing. ~2x faster than full IS. Use to filter candidates likely to
  miss the 0.5 floor before paying full backtest cost.
- **`python -m stratlab.arena.corr_check <strategy_path>`** — run the full
  IS backtest but compute ONLY max-corr-to-top-5 + loss-mode-corr.
  Doesn't write the leaderboard, doesn't consume an intent. Use to iterate
  signal mix until corr falls under 0.85 without burning intent slots.
- **`python -m stratlab.arena.corr_dump --top 12`** — pairwise IS-return
  Pearson matrix over the top-12 leaderboard rows. Use this for ensemble
  construction (opus-3 role) to pick components with all pairs <0.3.
- **`python -m stratlab.arena.dump_annual_calmar <strategy_id>`** — per-year
  return + Calmar from a strategy's persisted equity curve. Use to diagnose
  which years carry/break a strategy when h1/h2 columns don't tell the
  whole story.
- **`python -m stratlab.data.inception --tickers MTUM QUAL VIG --covers-is`** —
  multi-ticker cache-coverage check. Many factor ETFs (MTUM, QUAL launched
  2013; SCHD 2011) do NOT cover IS_START (2010). Run this BEFORE designing
  a strategy around those tickers to avoid silent fallbacks.

### Banned anti-patterns (will block on permission prompts)

The Bash validator can't statically analyze shell control flow, so even when each individual command is auto-allowed, wrapping them in a loop or substitution **fails the analysis and triggers a prompt**. Specifically:

❌ **`for` / `while` loops over file lists** — even if `head` is otherwise auto-allowed:
```
# BANNED — fails static analysis even though head is auto-allowed
for f in data/market/etfs/single_stock_leveraged/{TSLL,NVDL,AAPB}_1d.csv; do
  echo -n "$f: "; head -2 "$f" | tail -1 | cut -d, -f1
done
```

✅ **Use the Read tool, one file at a time** — Read calls don't prompt and you can call it 5 times in parallel:
```
# Five parallel Read tool calls in a single message — all auto-allowed.
Read(file_path="data/market/etfs/single_stock_leveraged/TSLL_1d.csv", limit=2)
Read(file_path="data/market/etfs/single_stock_leveraged/NVDL_1d.csv", limit=2)
Read(file_path="data/market/etfs/single_stock_leveraged/AAPB_1d.csv", limit=2)
Read(file_path="data/market/etfs/single_stock_leveraged/MSFU_1d.csv", limit=2)
Read(file_path="data/market/etfs/single_stock_leveraged/CONL_1d.csv", limit=2)
```

❌ **`python -c "..."` / `python3 -c "..."`** — arbitrary code execution; intentionally NOT allowlisted. If you want to compute something, call an existing harness CLI; if none exists, append a wishlist entry and use what's available.

❌ **Command substitution / pipes that hide arguments**: `$(...)`, backticks, complex pipelines through awk/sed for filtering. Auto-validation rejects these.

❌ **Glob expansion in Bash arguments** like `head data/market/*/TSLL*` — same static-analysis failure as for-loops.

### When the existing CLI doesn't quite cover what you need

Don't improvise a workaround. Append a wishlist entry naming the gap concretely (e.g., *"WANT: `--tickers TSLL NVDL AAPB` flag on `stratlab.data.inception` for ad-hoc multi-ticker queries"*) and proceed with what's available — typically 3-5 `Read` tool calls in parallel will get you the same data without prompts.

---

## Hard rule for ALL subagents: do not improvise infrastructure

Subagents stay inside the assigned scope: read the harness, write strategy files, submit. They do **NOT**, in any phase or role:

- add helper functions to existing modules (`stratlab/analytics/`, `stratlab/data/`, `stratlab/engine/`, `stratlab/arena/`, etc.)
- write new CLI scripts or change existing ones
- modify harness rules (gates, windows, corr threshold, paths)
- introduce a new config knob, env var, or settings field
- refactor adjacent code that wasn't broken
- create their own metric / tearsheet / utility / data-loader module
- pip install new dependencies
- touch `pyproject.toml`, `requirements*.txt`, or anything under `tests/`

If a subagent **feels something is missing** — a metric, a helper, a different filter, a tweak to the harness, anything — the action is to **append a one-line wish to `tmp/arena/wishlist.md`** and move on. The orchestrator triages between rounds; the round itself ships only what the existing harness can do today.

### Wishlist format (append-only, one wish per line)

```
- [<agent_id>, <gen_N>] WANT: <terse description>. WHY: <one-sentence concrete trigger>. KIND: helper|metric|harness-rule|cli|doc|other.
```

Concrete examples:

```
- [sonnet-3, gen_4] WANT: a `rolling_zscore(series, lookback)` helper in stratlab.analytics. WHY: reimplemented inline 4 times across my 3 strategies. KIND: helper.
- [opus-3, gen_4] WANT: pairwise IS-correlation matrix CLI dump. WHY: had to reimplement reading returns_matrix.csv + pearson + symbol-pair iteration to find low-corr triplets. KIND: cli.
- [opus-4, gen_4] WANT: optional `--commission-pct` override on submit.py for stress runs. WHY: reran 3 strategies by editing source manually to test 25 bps cost; risky and not reproducible. KIND: harness-rule.
```

The orchestrator reads `wishlist.md` during Step 4 and surfaces top recurring requests in the round memo. The user (not subagents) decides what gets implemented and when.

---

## Step 0 — Pre-flight

Run these checks before spawning anything. If any fail, stop and tell the user.

1. Imports clean: `.venv/bin/python -c "import yfinance, curl_cffi"`
2. Catalog current: `.venv/bin/python -m stratlab.data.catalog show` (if stocks-by-sector looks empty, run `rebuild` first)
3. Tests green: `.venv/bin/pytest tests/ -x -q` (if anything fails the harness state is unsafe to run on)
4. **IS regime fingerprint**: `.venv/bin/python -m stratlab.arena.regime_check --window is` — surface top-2-year concentration and VIX-regime composition before spawning. If it warns about >50% top-2y concentration or extreme VIX skew, paste the warning into the round's `phase2_brief.md` later and tell the user; agents should know the IS window is biased so they don't over-interpret headline Calmar
5. Pick `gen_N` = lowest unused integer under `stratlab/strategies/arena/`
6. Create `stratlab/strategies/arena/<gen_N>/` and `tmp/arena/round_<gen_N>/`
7. Ensure `tmp/arena/wishlist.md` exists (create with a one-line header if missing — `# Arena wishlist (cross-round, append-only)`); subagents will append here whenever they feel infrastructure is missing
8. If `wipe_leaderboard`: `mv tmp/arena tmp/arena_archive/<UTC-timestamp>/` and recreate `tmp/arena/` + the wishlist file. Otherwise leave the existing leaderboard intact (corr filter benefits from prior rows)

---

## Step 1 — Phase 1 (Sonnet generation, parallel)

Spawn `phase1_generators` Sonnet subagents in a **single message with multiple Agent tool uses** so they run concurrently. Use `subagent_type=general-purpose`, `model=sonnet`. Each gets the prompt below with `{i}`, `{gen_N}`, and `{phase1_ideas_per_agent}` substituted.

### Sonnet generator prompt template

```
You are sonnet-{i} of {phase1_generators}, a strategy generator in round-{gen_N}.

Mission: produce {phase1_ideas_per_agent} distinct trading hypotheses,
implement each as a strategy file, and submit each via the harness. Each
accepted submission earns one leaderboard row.

Read first (do not skip):
  - stratlab/arena/AGENT_PLAYBOOK.md (this playbook)
  - stratlab/arena/config.py            (IS/OOS windows, gates, paths)
  - stratlab/arena/intents.csv          (claimed hypotheses — don't dupe)
  - tmp/arena/leaderboard.csv           (existing strategies — diversify)
  - stratlab/strategies/base.py         (Strategy ABC — what to subclass)
  - stratlab/strategies/curated/        (CANONICAL reference set: 5 OOS-validated
                                         strategies covering xsect-momentum,
                                         VIX-gated, dip-buy, ensemble. Read these
                                         when you want to see how a working
                                         strategy is structured AND what's
                                         already proven, in one place. Don't
                                         dupe them — diversify against them.)
  - stratlab/engine/context.py          (BarContext API: ctx.symbols,
                                         ctx.signal_symbols, ctx.history,
                                         ctx.closes, ctx.closes_window)
  - stratlab/engine/broker.py           (is_tradeable_symbol — ^X, =F, =X are
                                         signal-only; broker rejects orders)

Hypothesis pre-commit (PREVENTS PARALLEL DUPLICATION):
  python -m stratlab.arena.intents commit \
      --agent sonnet-{i} --generation {gen_N} \
      --hypothesis "<one-line, specific>"
  # fcntl.flock'd internally — concurrent agents serialize cleanly. Commit
  # ALL {phase1_ideas_per_agent} ideas before implementing any of them, so
  # other agents see your claims while you code.

For each committed hypothesis:
  1. Implement at stratlab/strategies/arena/{gen_N}/<slug>.py.
     Module must export STRATEGY = MyStrategy() and may set UNIVERSE
     = "sp500" | "popular_etfs" | list[str]. Default is sp500 if omitted.
  2. Submit:
       python -m stratlab.arena.submit stratlab/strategies/arena/{gen_N}/<slug>.py
     Exit codes: 0=accepted, 2=n_trades<50, 3=corr>0.85 vs top-5,
                 4=runtime error, 5=IS Calmar<0.5
     The harness auto-marks the intent submitted (on 0) or abandoned (on 2-5).
  3. If rejected, append a 2-line postmortem to tmp/arena/dead_ends.md:
     {hypothesis | reason | exit_code | round}.

Hard constraints (enforced by harness; trying to bypass is a waste):
  - allow_short=False, enforce_cash=True, initial_cash=100_000
  - IS window only — DO NOT read OOS data; promote.py is the sole reader
  - Don't edit submit.py, config.py, leaderboard.csv, or intents.csv directly
  - Index levels / futures / FX are SIGNAL-ONLY (broker rejects orders).
    Read them via ctx.history("^VIX") for regime/macro signals.

Improvise NOTHING outside your strategy file. If you feel a helper, metric,
filter, harness tweak, or anything else is missing, append ONE LINE to
tmp/arena/wishlist.md in this format:
  - [sonnet-{i}, gen_{gen_N}] WANT: <thing>. WHY: <concrete trigger>. KIND: helper|metric|harness-rule|cli|doc|other.
Then keep going with what the harness can do today. Do NOT add helpers to
stratlab/analytics, stratlab/data, stratlab/engine, or anywhere else. Do NOT
modify the harness, tests, pyproject.toml, or any module that isn't your
strategy file under stratlab/strategies/arena/{gen_N}/.

Stop when ANY of:
  - {phase1_ideas_per_agent} accepted submissions, OR
  - 6 distinct hypotheses attempted (some rejected), OR
  - 30 min wall-clock

Report back: one paragraph per hypothesis covering
  {idea, IS Calmar, n_trades, Sharpe, max_dd, accepted|reason}.
Keep total reply under 600 words.
```

While Phase 1 runs, the orchestrator does nothing — wait for all subagents to return. Don't poll filesystem state; trust the task notifications.

---

## Step 2 — Mid-round handoff (orchestrator only)

After all Phase 1 agents return, before spawning Phase 2:

1. Read leaderboard: `cat tmp/arena/leaderboard.csv` — filter to rows where `generation == gen_N`
2. Cluster survivors by theme (e.g. *vol-regime*, *sector-rotation*, *mean-reversion*, *event-driven*, *cross-asset*, *breadth*, *factor-tilt*). Don't be precise — 3-6 cluster labels is enough
3. Read `tmp/arena/dead_ends.md`; flag themes where ≥3 attempts failed (saturated — Phase 2 should NOT mine those further)
4. Build a "Phase 2 brief" containing:
   - top 8-12 leaderboard rows (strategy_id, theme tag, IS Calmar, n_trades, brief description)
   - 3-5 dead-end clusters with attempt counts
   - any cross-cutting observation (e.g. "all FOMC ideas died at the corr gate" or "breadth signals dominate the top 5")
5. Save to `tmp/arena/round_<gen_N>/phase2_brief.md` so Opus subagents can re-read it

---

## Step 3 — Phase 2 (Opus refinement, 5 fixed roles, parallel)

Spawn 5 Opus subagents in **a single message with 5 Agent tool uses**. Use `subagent_type=general-purpose`, `model=opus`. Roles are pre-assigned by the orchestrator — do NOT let agents pick.

| ID | Role | Job |
|---|---|---|
| opus-1 | `best_of_theme` | Pick the top 1 strategy per theme cluster. Mutate each — different lookback, different sizing rule, different entry trigger — into a variant the 0.85 corr filter still admits |
| opus-2 | `gap_finder` | Scan the leaderboard for thematic gaps (asset classes / regimes / horizons not represented). Propose ≤5 strategies that fill specific gaps |
| opus-3 | `ensemble` | Combine 2-3 low-correlation survivors (compute pairwise IS-return correlations from `returns_matrix.csv`; pick triplets with all pairs <0.3) into a single ensemble strategy file (weighted, voting, or regime-gated). Backtest the ensemble before submitting |
| opus-4 | `critic` | Pick top 3 survivors. For each, try to break it: sub-period stability (split IS into halves, see if Calmar holds), transaction-cost stress (re-run with 25 bps commission), regime dependency (does it survive 2010-14 vs 2015-18?). Write findings to `tmp/arena/round_<gen_N>/critique.md` instead of submitting strategies |
| opus-5 | `wildcard` | Explicit anti-consensus. Pick one hypothesis no Phase 1 agent touched (use a non-traditional signal, e.g. ^MOVE/^SKEW regime, breadth thrust, single-stock leveraged ETF rotation). High variance OK — better to fail interestingly than succeed boringly |

### Opus refiner prompt template (substitute `{role}` and `{role_specifics}`)

```
You are opus-{i} ({role}), Phase 2 refiner in round-{gen_N}.

Read first:
  - tmp/arena/round_{gen_N}/phase2_brief.md   (your Phase 1 summary)
  - stratlab/arena/AGENT_PLAYBOOK.md          (this playbook)
  - tmp/arena/leaderboard.csv                 (full survivor set)
  - stratlab/strategies/curated/              (canonical reference: OOS-validated
                                               strategies — read these to see
                                               what's structurally working before
                                               proposing variants)
  - tmp/arena/returns_matrix.csv              (only opus-3 needs this for corrs)

Role-specific instructions:
{role_specifics}

Same harness rules as Phase 1:
  - hypothesis pre-commit before implementing
  - submit via python -m stratlab.arena.submit
  - same exit codes, same OOS isolation, same gen_{gen_N}/ directory
  - allow_short=False, enforce_cash=True, IS only

Improvise NOTHING outside your strategy file. If you feel a helper, metric,
filter, harness tweak, or anything else is missing, append ONE LINE to
tmp/arena/wishlist.md in this format:
  - [opus-{i}, gen_{gen_N}] WANT: <thing>. WHY: <concrete trigger>. KIND: helper|metric|harness-rule|cli|doc|other.
Then proceed with what the harness can do today. Do NOT add helpers to
stratlab/analytics, stratlab/data, stratlab/engine, or anywhere else. Do NOT
modify the harness, tests, pyproject.toml, or any module that isn't your
strategy file under stratlab/strategies/arena/{gen_N}/. The orchestrator
triages wishes between rounds — your job is to ship strategies, not infra.

Output:
  - For opus-1, opus-2, opus-3, opus-5: 1 strategy file per accepted submission
  - For opus-4: tmp/arena/round_{gen_N}/critique.md (no strategy files)

Stop when role goal met or 45 min wall-clock. Report under 800 words.
```

---

## Step 4 — Round wrap-up (orchestrator)

After all Phase 2 agents return:

1. OOS-evaluate the top survivors:
   ```
   python -m stratlab.arena.promote --top {oos_promote_top}
   ```
2. Read the updated leaderboard. Note IS-Calmar → OOS-Calmar deltas; flag any strategy with OOS Calmar < 0.3 of IS Calmar (severe overfit) or OOS_Calmar < 0 (broken on OOS).
3. Read `tmp/arena/wishlist.md`; collect entries tagged with this round's `gen_N`. Cluster duplicates (multiple agents asking for the same thing = real signal). Don't implement anything — the user triages.
4. Write `tmp/arena/round_<gen_N>/memo.md` containing:
   - **Top 5 by IS Calmar** with OOS Calmar comparison
   - **Best OOS performers** (different ranking — these are the ones to actually trust)
   - **Theme winners and losers** for this round
   - **Critic highlights** (paste 3-5 bullets from `critique.md`)
   - **Ensemble result** (whether opus-3's ensemble beat its components)
   - **Wildcard outcome** (often rejected — note it either way; the negative result is data)
   - **Wishlist deltas this round** — bulleted list of new wishes from `wishlist.md` grouped by KIND, with duplicate-counts (e.g. *"3× WANT: rolling_zscore helper"*). One-line each
5. Tell the user: memo path, leaderboard delta (`+N rows, top mover X`), any flagged regressions, and the count of new wishlist entries (so they know whether to triage now or later).

The memo also includes a **"Killed for timeout"** section listing any subagents the orchestrator TaskStop'd during the round (see Timeouts below). One-line each: `agent_id, role, elapsed, what-it-had-when-killed`. This is signal — recurring kills on the same role mean the budget for that role is too tight or the work is structurally too heavy.

---

## Timeouts

Each subagent has a self-policed wall-clock **budget** plus an orchestrator-enforced **hard cap**. The budget is what the agent should target; the cap is when the orchestrator kills the run.

| Phase | Self-policed budget | Orchestrator hard cap |
|---|---|---|
| Phase 1 sonnet generator | 30 min | 45 min |
| Phase 2 opus refiner | 45 min | 60 min |

### Self-policing (encoded in the prompts; agent's responsibility)

- Before starting another long backtest, check elapsed time.
- Prefer narrow universes (`UNIVERSE = "sp500"` or `"popular_etfs"`) when a strategy doesn't need all 4,982 tickers. Full-universe cross-sectional runs cost 5-10 min each and burn the budget fast.
- If you've used 80% of budget with fewer accepted submissions than your target, **stop and report what you have**. A partial round of 2 accepted strategies + a clean exit beats a forced kill at the hard cap with 0 surfaced output.

### Orchestrator enforcement (my responsibility)

- I track elapsed time per spawned agent via the task system. I do not poll the filesystem; I rely on the Agent tool's task notifications.
- At the hard cap, I issue `TaskStop` against the agent.
- The subagent's last partial output is still readable from the task log — I scrape it for any accepted submissions (already on the leaderboard, so safe) and any wishlist entries (already in `wishlist.md`, so safe). Anything mid-implementation is lost.
- After a kill, I run `python -m stratlab.arena.intents mark <intent_id> abandoned` for any of that agent's intents still in `committed` state, so the registry doesn't accumulate ghosts.
- The kill is logged into the round memo's `Killed for timeout` section (Step 4 above) so the user can see exactly which roles ran over budget.

### What does NOT exist today

- **In-harness per-backtest timeout.** A single `submit.py` invocation that starts a multi-hour backtest will run to completion — there's no `--timeout-seconds` in the submit CLI. The only enforcement is the orchestrator killing the entire subagent. If this becomes a recurring failure mode (especially on Phase 2 ensembles or wide-universe ranking runs), file it as a wishlist entry (`KIND: harness-rule`) so the user can implement a real per-backtest timeout in `submit.py` — that's the right fix, not a workaround inside a strategy file.



Do NOT trigger another generation round automatically. The user decides.

---

## Operational gotchas (lessons from prior rounds — do not relearn)

- **`fewer-permission-prompts` skill auto-derails subagents.** The narrow allowlist in `.claude/settings.json` is what prevents this. Subagents inherit `settings.json` (project-shared) but NOT `settings.local.json` (user-local). If a subagent's transcript shows it triggered the skill instead of doing the task, that work is lost — respawn it. Do not widen the allowlist; the fix was making it narrow.

- **Intent registry races.** `intents.csv` is `fcntl.flock`'d internally. Subagents that bypass `python -m stratlab.arena.intents` and edit the CSV by hand will corrupt it. If an agent crashes mid-implementation, its intent stays in `committed` state — clear via `python -m stratlab.arena.intents mark <intent_id> abandoned`.

- **Long backtests on 4000-ticker universes.** A cross-sectional ranking strategy over the full default_universe can take 5+ minutes. If a sonnet appears stuck for >10 min on a single backtest, TaskStop it and respawn with `UNIVERSE = "sp500"` or `UNIVERSE = "popular_etfs"`. Most cross-sectional ideas don't need 4000 names to differentiate.

- **`gen_N` collision.** If `stratlab/strategies/arena/gen_<N>/` already has files, pick `<N+1>` — never overwrite a prior round's files.

- **OOS leakage.** Only `promote.py` reads OOS. `submit.py` and all generators must not. The harness enforces this with separate code paths; agents that try to load OOS data via `load_universe(start=OOS_START)` will get caught at code review (and would be wasted work — the leaderboard only accepts IS metrics from `submit.py`).

- **Phase 2 role assignment is fixed.** Don't let opus subagents pick their own role — they all converge on `gap_finder` because it's the easiest. The orchestrator hands each subagent its role in the spawn prompt.

- **Universe filtering.** Use `filter_universe_by_window_overlap(tickers, start, end)` for any backtest. The older `filter_universe_by_inception` is too strict — it excludes mid-window IPOs (UBER from a 2019-onwards OOS run). Submit and promote already use the correct filter; strategy code that pre-filters its own universe must use the overlap version.

- **Tradability.** Index levels (`^VIX`, `^TNX`), continuous futures (`ES=F`, `GC=F`), and spot FX (`EURUSD=X`) are signal-only — `is_tradeable_symbol()` returns False, broker rejects orders. Strategies can read them via `ctx.history(sym)` or `ctx.signal_symbols`. Real exposure routes through ETFs (SPY, TLT, GLD, FXE, etc.) or single-stock leveraged ETFs (`stocks/single_stock_leveraged/`).

- **Cost-stress probes.** `submit.py` accepts `--commission-pct` and `--slippage-pct`. When either differs from the harness defaults (commission=0.001, slippage=0.0005), the run is treated as a probe — metrics print but the leaderboard is NOT updated. Use this to validate that a strategy's edge survives 25 bps higher round-trip costs without polluting the comparable-baseline leaderboard. Critic-role agents (opus-4) should use this instead of computing analytical cost-stress numbers.

- **Equity-curve dumps.** Every accepted submission writes a daily-equity CSV to `tmp/arena/equity_curves/<strategy_id>.csv`. `python -m stratlab.arena.dump_equity_curve <strategy_id>` reads it (or `--path` to print only the file path). Use this for sub-period drawdown analysis or any curve-level inspection that tearsheets don't expose cleanly.

- **Sub-period and loss-mode metrics on the leaderboard.** Each row now carries `is_calmar_h1`, `is_calmar_h2`, `is_calmar_min`, `is_pnl_top2y_pct` (regime-concentration metrics) and `loss_mode_corr_to_top5` (correlation on bottom-decile SPY days). Generators see these in the visible-columns set; agents should bias toward strategies whose `is_calmar_min` is close to `is_calmar` (stable across halves) and whose `loss_mode_corr_to_top5` is well below 0.85 (different loss-mode than existing leaders, even if daily-corr is similar). The harness does NOT gate on these — they're informational signals for ensemble construction and round-memo writing.

---

## Quick-invoke phrasings the user might use

- *"@stratlab/arena/AGENT_PLAYBOOK.md run"* — full round with defaults
- *"@stratlab/arena/AGENT_PLAYBOOK.md run with 5 sonnets × 3 ideas, skip Phase 2"* — Phase 1 only
- *"@stratlab/arena/AGENT_PLAYBOOK.md run gen_5, wipe leaderboard"* — fresh slate
- *"@stratlab/arena/AGENT_PLAYBOOK.md just step 4"* — re-run wrap-up only (e.g. after fixing a bug in `promote.py`)

The orchestrator interprets these against the parameters table at the top.
