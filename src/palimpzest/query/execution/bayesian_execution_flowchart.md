# Bayesian Execution Strategy

## Overview

The Bayesian Execution Strategy replaces the traditional Multi-Armed Bandit (MAB) UCB/LCB
approach with **Bayesian posterior sampling** and **Monte Carlo Pareto probability estimation**
to select the best physical operators for each logical operation in a query plan.

Each physical operator maintains **four probability distributions** that are continuously
updated as records are processed, enabling principled uncertainty-aware optimization.

---

## Probability Distributions per Physical Operator

Every physical operator tracks posteriors over four metrics:

| Metric | Distribution | Parameters | Interpretation |
|---|---|---|---|
| **Quality** | Beta(α, β) | α=1, β=1 (uniform prior) | Normalized output quality ∈ [0,1]. Higher is better. Updated with `α += quality`, `β += (1 − quality)`. |
| **Selectivity** | Beta(α, β) | α=1, β=1 (uniform prior) | Fraction of records that pass the operator ∈ [0,1]. Higher means less filtering. Updated with `α += passed`, `β += (1 − passed)`. |
| **Cost** | Normal(μ, σ) | μ=0.5, σ=0.3 | Monetary cost per record (USD). Lower is better. Updated via running mean `μ ← (1−1/n)μ + (1/n)·observed`. |
| **Time** | Normal(μ, σ) | μ=0.5, σ=0.3 | Execution time per record (seconds). Lower is better. Updated via running mean. |

> The **Beta distribution** is a conjugate prior for Bernoulli/binomial observations,
> making it ideal for bounded [0,1] metrics. The **Gaussian posterior** uses an
> incremental mean update for cost and time.


## Algorithm Phases (compact)

| Phase | What happens | Key detail |
|---|---|---|
| **1. Init** | One `BayesianOpFrontier` per logical op; each physical op gets 4 uninformative priors | `Beta(1,1)` for quality & selectivity; `Normal(0.5, 0.3)` for cost & time |
| **2. Select** | Pick the physical op with highest posterior mean quality: $\arg\max \frac{\alpha_q}{\alpha_q + \beta_q}$ | Greedy on quality; exploration comes later via Thompson sampling |
| **3. Execute & Score** | Run selected op on input records; score outputs via validator | Produces per-record stats: quality, selectivity (pass/fail), cost, time |
| **4. Update posteriors** | Feed observed averages into each op's four distributions | Beta: $\alpha' = \alpha + \bar{x},\; \beta' = \beta + (1{-}\bar{x})$. Gaussian: $\mu' = (1{-}\tfrac{1}{n})\mu + \tfrac{1}{n}x$ |
| **5. MC Pareto** | 40 rounds of Thompson sampling → Pareto front on sampled metrics → count frequency | Gives $P(\text{Pareto-optimal})$ per op; ops with $P < 5\%$ are demoted |
| **6. Reassign frontier** | Sort remaining ops by Pareto probability; top-$k$ become frontier, rest go to reservoir | Ensures only promising ops are actively explored |
| **7. Loop / Stop** | Repeat from phase 2 until budget exhausted **or** 5 consecutive stall iterations | Stall = zero new samples in one full pass |

---

## Key Advantages over MAB (UCB/LCB)

| Aspect | MAB (UCB/LCB) | Bayesian (This Strategy) |
|---|---|---|
| **Uncertainty model** | Confidence intervals (frequentist) | Full posterior distributions |
| **Exploration** | Upper/lower confidence bounds | Thompson sampling (natural exploration) |
| **Multi-objective** | Scalarized or single-metric | True Pareto front over 4 metrics |
| **Frontier selection** | Deterministic bounds | Probabilistic Pareto membership |
| **Adaptivity** | Slow convergence with sparse data | Informative even with few observations |

---

## Pseudocode

The algorithm is composed of **5 functions**. The table below summarizes each one, followed by the full pseudocode.

| Function | Purpose |
|---|---|
| `BetaPosterior` | Conjugate Beta model for bounded [0,1] metrics (quality, selectivity). `update` shifts the distribution toward observed values; `sample` draws a random plausible value (Thompson sampling); `mean` returns the current expected value. |
| `GaussianPosterior` | Simplified Normal model for unbounded positive metrics (cost, time). `update` computes an incremental running mean; `sample` draws a noisy estimate around the current mean. |
| `Initialize` | Creates one `BayesianOpFrontier` per logical operator in the plan. Each physical operator receives four independent prior distributions (2 Beta + 2 Gaussian). The first $k$ operators form the initial frontier; the rest go to a reservoir. |
| `Execute` | Main loop. Iterates until the sample budget is spent. Each iteration: selects the best operator per logical op (highest posterior mean quality), executes it, scores outputs, updates plan stats, feeds results downstream, then calls `update_frontier` to revise operator rankings. Stops early after 5 stalled iterations. |
| `update_frontier` | **Core Bayesian logic.** (A) Aggregates per-record observations into means. (B) Updates the four posteriors for each observed operator. (C) Runs 40 Monte Carlo rounds of Thompson sampling + Pareto dominance to estimate $P(\text{Pareto-optimal})$ per operator. (D) Keeps operators with $P \geq 5\%$; top-$k$ by probability become the active frontier, the rest go to the reservoir. |
| `DOMINATES` | Multi-objective dominance test over 4 metrics. Returns true when operator A is at least as good as B on every metric and strictly better on at least one (minimize cost/time, maximize quality/selectivity). |

```text
════════════════════════════════════════════════════════════════════
  BAYESIAN EXECUTION STRATEGY — PSEUDOCODE
════════════════════════════════════════════════════════════════════

── Posterior Models ──────────────────────────────────────────────

  BetaPosterior(α=1, β=1):
  │ Tracks a [0,1] metric via Beta-Bernoulli conjugacy.
  │ Starts as uniform; concentrates around the true rate
  │ as more observations arrive.
  │
  │  update(x):   α ← α + x;  β ← β + (1 − x)
  │  sample():    return ~ Beta(α, β)
  │  mean:        return α / (α + β)

  GaussianPosterior(μ=0.5, σ=0.3, n=1):
  │ Tracks a positive continuous metric via incremental
  │ running mean. σ stays fixed (simplified).
  │
  │  update(x):   n ← n + 1;  μ ← (1 − 1/n)·μ + (1/n)·x
  │  sample():    return ~ Normal(μ, σ)

── Initialize ────────────────────────────────────────────────────
  Sets up one Bayesian frontier per logical operator, each
  containing the candidate physical ops with fresh priors.

  FUNCTION Initialize(plan, dataset, k, j):
      plan_stats ← new PlanStats(plan)
      shuffled_indices ← shuffle(dataset)

      FOR EACH logical_op IN plan (topological order):
          op_set ← physical operators for logical_op
          FOR EACH op IN op_set:
              posteriors[op] ← {
                  quality      : BetaPosterior(1, 1),
                  selectivity  : BetaPosterior(1, 1),
                  cost         : GaussianPosterior(0.5, 0.3),
                  time         : GaussianPosterior(0.5, 0.3)
              }
          frontier_ops  ← op_set[1..k]
          reservoir_ops ← op_set[k+1..]

      RETURN op_frontiers, plan_stats

── Execute (main loop) ──────────────────────────────────────────
  Iterates over records, executing operators, scoring results,
  and refining posterior beliefs until the budget is exhausted.

  FUNCTION Execute(plan, dataset, validator, sample_budget):
      op_frontiers, plan_stats ← Initialize(plan, dataset, k, j)
      samples_drawn ← 0
      stall_count   ← 0

      WHILE samples_drawn < sample_budget:
          needed_indices ← ∪ frontier.next_indices()  ∀ frontier
          prev ← samples_drawn

          FOR EACH logical_op IN plan (topological order):
              fr ← op_frontiers[logical_op]

              // 1) SELECT: pick op with highest posterior mean quality
              best_op ← argmax  α_q/(α_q+β_q)   over fr.frontier_ops

              // 2) EXECUTE: run the selected operator on input records
              inputs ← fr.get_inputs(needed_indices, best_op)
              IF inputs = ∅: BREAK
              outputs, n ← execute(logical_op, inputs)
              samples_drawn += n

              // 3) SCORE: evaluate output quality via the validator
              scored ← validator.score(outputs)

              // 4) STATS: accumulate per-record metrics in plan_stats
              plan_stats.add(logical_op, outputs.record_stats)

              // 5) FEED DOWNSTREAM: pass results to the next operator
              next ← plan.get_next(logical_op)
              IF next ≠ NULL:
                  op_frontiers[next].update_inputs(scored)

              // 6) UPDATE FRONTIER: revise posteriors + Pareto ranking
              fr.update_frontier(plan_stats)

          // Stall detection (no new samples → stall)
          IF samples_drawn = prev:
              stall_count += 1
              IF stall_count ≥ 5: BREAK
          ELSE:
              stall_count ← 0

      RETURN plan_stats

── update_frontier (core Bayesian logic) ────────────────────────
  Called after each logical operator is executed. Updates
  posterior beliefs, then uses Monte Carlo simulation to
  re-estimate which operators belong on the Pareto frontier.

  FUNCTION update_frontier(plan_stats):

      // PHASE A — Aggregate observations
      FOR EACH op WITH new records:
          avg_q ← mean(quality),  avg_s ← mean(selectivity)
          avg_c ← mean(cost),     avg_t ← mean(time)

      // PHASE B — Update the four posterior distributions
      //   Each update shifts the distribution closer to the
      //   observed performance of that physical operator.
      FOR EACH op WITH observations:
          posteriors[op].quality.update(avg_q)
          posteriors[op].selectivity.update(avg_s)
          posteriors[op].cost.update(avg_c)
          posteriors[op].time.update(avg_t)

      // PHASE C — Monte Carlo Pareto estimation
      //   Repeatedly sample plausible metrics from each op's
      //   posteriors, compute the Pareto front on those samples,
      //   and count how often each op appears on it.
      pool ← frontier_ops ∪ reservoir_ops
      pareto_count[op] ← 0   ∀ op ∈ pool

      REPEAT 40 times:
          FOR EACH op ∈ pool:              // Thompson sampling
              s[op].q ← posteriors[op].quality.sample()
              s[op].s ← posteriors[op].selectivity.sample()
              s[op].c ← posteriors[op].cost.sample()
              s[op].t ← posteriors[op].time.sample()

          pareto_set ← ∅
          FOR EACH opA ∈ pool:
              IF ∄ opB ∈ pool s.t. DOMINATES(s[opB], s[opA]):
                  pareto_set ← pareto_set ∪ {opA}

          FOR EACH op ∈ pareto_set:
              pareto_count[op] += 1

      P[op] ← pareto_count[op] / 40       // Pareto probability

      // PHASE D — Reassign frontier
      //   Only operators with ≥5% Pareto probability survive.
      //   The top-k by probability form the active frontier.
      candidates ← { op : P[op] ≥ 0.05 }
      candidates ← sort by P[op] descending
      frontier_ops  ← candidates[1..k]
      reservoir_ops ← everyone else

── DOMINATES ────────────────────────────────────────────────────
  Standard Pareto dominance over 4 objectives.
  Returns TRUE iff A is ≥ B on all metrics and > on at least one.
  (minimize cost & time, maximize quality & selectivity)

  FUNCTION DOMINATES(A, B):
      at_least ← A.cost ≤ B.cost  ∧  A.time ≤ B.time
                 ∧  A.quality ≥ B.quality  ∧  A.selectivity ≥ B.selectivity
      strict   ← A.cost < B.cost  ∨  A.time < B.time
                 ∨  A.quality > B.quality  ∨  A.selectivity > B.selectivity
      RETURN at_least ∧ strict
```
