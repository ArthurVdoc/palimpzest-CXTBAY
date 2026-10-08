import logging
import numpy as np
from collections import defaultdict

from palimpzest.query.execution.execution_strategy import SentinelExecutionStrategy
from palimpzest.query.execution.mab_execution_strategy import OpFrontier
from palimpzest.core.models import SentinelPlanStats
from palimpzest.utils.progress import create_progress_manager
from palimpzest.query.operators.scan import ScanPhysicalOp, ContextScanOp
from palimpzest.query.operators.join import JoinOp


logger = logging.getLogger(__name__)


# baynesian implementation: simple conjugate models

class BetaPosterior:
    """Posterior para métricas em [0,1], ex.: qualidade normalizada, seletividade."""
    def __init__(self, a=1.0, b=1.0):
        self.a = a
        self.b = b

    def update(self, value):
        # value em [0,1]
        self.a += value
        self.b += 1 - value

    def sample(self):
        return np.random.beta(self.a, self.b)

    @property
    def mean(self):
        total = self.a + self.b
        if total == 0:
            return 0.5
        return self.a / total


class GaussianPosterior:
    """Normal posterior with known variance (simplified).
    We treat cost and time as positive continuous variables."""
    def __init__(self, mu=1.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma
        self.n = 1

    def update(self, value):
        self.n += 1
        lr = 1 / self.n
        self.mu = (1 - lr) * self.mu + lr * value

    def sample(self):
        return np.random.normal(self.mu, self.sigma)


# baynesian operator frontier with posterior tracking

class BayesianOpFrontier(OpFrontier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # posterior distributions per operator
        self.posteriors = {}
        for op in self.frontier_ops + self.reservoir_ops:
            full_id = op.get_full_op_id()
            self.posteriors[full_id] = {
                "quality": BetaPosterior(),
                "selectivity": BetaPosterior(),
                "cost": GaussianPosterior(mu=0.5, sigma=0.3),
                "time": GaussianPosterior(mu=0.5, sigma=0.3)
            }

    def get_pool_size(self):
        """Return the total number of operators in the pool (frontier + reservoir + off-frontier)."""
        return len(self.frontier_ops) + len(self.reservoir_ops) + len(self.off_frontier_ops)

    # baynesian update of frontier, replacing UCB/LCB logic

    def update_frontier(self, unique_logical_op_id, plan_stats: SentinelPlanStats):
        """
        Replaces the UCB/LCB approach with Bayesian posterior sampling
        + Monte Carlo Pareto probability estimation.
        """

        # agregate record stats for operators in this frontier
        full_op_id_to_op_stats = plan_stats.operator_stats.get(unique_logical_op_id, {})

        # accumulate means from the new record stats
        means = defaultdict(lambda: defaultdict(float))
        counts = defaultdict(int)

        for full_op_id, op_stats in full_op_id_to_op_stats.items():
            for rec in op_stats.record_op_stats_lst:
                counts[full_op_id] += 1
                means[full_op_id]["quality"] += rec.quality or 0
                means[full_op_id]["selectivity"] += rec.passed_operator
                means[full_op_id]["cost"] += rec.cost_per_record
                means[full_op_id]["time"] += rec.time_per_record

        # update baynesian posteriors
        for full_op_id, cnt in counts.items():
            if cnt == 0:
                continue

            post = self.posteriors[full_op_id]
            m = means[full_op_id]

            post["quality"].update(m["quality"] / max(cnt, 1))
            post["selectivity"].update(m["selectivity"] / max(cnt, 1))
            post["cost"].update(m["cost"] / max(cnt, 1))
            post["time"].update(m["time"] / max(cnt, 1))

        # baynesian pareto frontier estimation

        def sample_operator_metrics(op):
            """Draw a Thompson sample for each metric."""
            fid = op.get_full_op_id()
            p = self.posteriors[fid]
            return {
                "quality": p["quality"].sample(),
                "selectivity": p["selectivity"].sample(),
                "cost": p["cost"].sample(),
                "time": p["time"].sample()
            }

        def dominates(opA, opB):
            """Strict Bayesian dominance test (minimizing cost/time,
            maximizing quality/selectivity)."""
            A = opA
            B = opB

            better_or_equal = (
                (A["cost"] <= B["cost"]) and
                (A["time"] <= B["time"]) and
                (A["quality"] >= B["quality"]) and
                (A["selectivity"] >= B["selectivity"])
            )

            strictly_better = (
                (A["cost"] < B["cost"]) or
                (A["time"] < B["time"]) or
                (A["quality"] > B["quality"]) or
                (A["selectivity"] > B["selectivity"])
            )

            return better_or_equal and strictly_better

        # monte carlo estimation of pareto frontier probabilities (ver essa parte em mais detalhes)
        MC_SAMPLES = 200
        op_list = self.frontier_ops + self.reservoir_ops
        frontier_prob = {op.get_full_op_id(): 0 for op in op_list}

        for _ in range(MC_SAMPLES):
            samples = {op.get_full_op_id(): sample_operator_metrics(op) for op in op_list}

            # compute pareto set on this sample
            pareto = set()
            for opA in op_list:
                A = samples[opA.get_full_op_id()]
                dominated = False
                for opB in op_list:
                    if opA is opB:
                        continue
                    B = samples[opB.get_full_op_id()]
                    if dominates(B, A):  # B dominates A
                        dominated = True
                        break
                if not dominated:
                    pareto.add(opA.get_full_op_id())

            # add to probability counts
            for fid in pareto:
                frontier_prob[fid] += 1

        # normalize
        for fid in frontier_prob:
            frontier_prob[fid] /= MC_SAMPLES

        #Keep only operators with frontier probability above threshold
        THRESH = 0.05  # 5% probability of being Pareto-optimal
        new_frontier = []

        for op in op_list:
            fid = op.get_full_op_id()
            if frontier_prob[fid] >= THRESH:
                new_frontier.append(op)

        # baynesian update of frontier, reservoir, and off-frontier lists
        # keep same size k
        new_frontier = sorted(
            new_frontier,
            key=lambda op: frontier_prob[op.get_full_op_id()],
            reverse=True
        )

        # limit frontier to k ops (policy parameter)
        self.frontier_ops = new_frontier[: self.k]

        # reservoir = the rest
        remaining = [op for op in op_list if op not in self.frontier_ops]
        self.reservoir_ops = remaining
        self.off_frontier_ops = []

        logger.info(
            f"[Bayesian Frontier] Operators kept: {[op.get_op_id() for op in self.frontier_ops]}"
        )


# execution strategy: mininum override

class BayesianExecutionStrategy(SentinelExecutionStrategy):

    def execute_sentinel_plan(self, plan, train_dataset, validator):
        logger.info(f"Executing (Bayesian) plan {plan.plan_id}")

        # initialize stats
        plan_stats = SentinelPlanStats.from_plan(plan)
        plan_stats.start()

        # shuffle dataset indices (same as MAB version)
        dataset_id_to_shuffled_source_indices = {}
        for dataset_id, dataset in train_dataset.items():
            idxs = [f"{dataset_id}-{i}" for i in range(len(dataset))]
            self.rng.shuffle(idxs)
            dataset_id_to_shuffled_source_indices[dataset_id] = idxs

        # create Bayesian frontiers instead of MAB frontiers
        op_frontiers = {}
        for topo_idx, (logical_op_id, op_set) in enumerate(plan):
            unique_id = f"{topo_idx}-{logical_op_id}"
            source_unique_ids = plan.get_source_unique_logical_op_ids(unique_id)
            root_ids = plan.get_root_dataset_ids(unique_id)

            sample_op = op_set[0]

            if isinstance(sample_op, (ScanPhysicalOp, ContextScanOp)):
                assert len(root_ids) == 1, f"Scan for {sample_op} has {len(root_ids)} > 1 root datasets"
                source_idxs = dataset_id_to_shuffled_source_indices[root_ids[0]]

            elif isinstance(sample_op, JoinOp):
                assert len(source_unique_ids) == 2, f"Join for {sample_op} has {len(source_unique_ids)} != 2"
                left = op_frontiers[source_unique_ids[0]].source_indices
                right = op_frontiers[source_unique_ids[1]].source_indices
                source_idxs = [(l, r) for l in left for r in right]

            else:
                # "normal" operator (filter/map/convert)
                assert len(source_unique_ids) == 1, f"Unexpected multi-source operator: {sample_op}"
                source_idxs = op_frontiers[source_unique_ids[0]].source_indices

            op_frontiers[unique_id] = BayesianOpFrontier(
                op_set,
                source_unique_ids,
                root_ids,
                source_idxs,
                self.k,
                self.j,
                self.seed,
                self.policy,
                self.priors,
            )

        # start progress manager
        self.progress_manager = create_progress_manager(
            plan, sample_budget=self.sample_budget, progress=self.progress
        )
        self.progress_manager.start()

        try:
            # identical loop to MAB, except frontiers use Bayesian logic
            samples_drawn = 0
            consecutive_no_progress_iterations = 0
            max_no_progress_iterations = 5  # Stop if no progress for 5 iterations
            
            while samples_drawn < self.sample_budget:
                # collect needed indices
                source_indices_to_sample = set()
                for fr in op_frontiers.values():
                    source_indices_to_sample |= fr.get_source_indices_for_next_iteration()

                # run operators in topological order
                samples_before_iteration = samples_drawn
                for topo_idx, (logical_op_id, _) in enumerate(plan):
                    unique_id = f"{topo_idx}-{logical_op_id}"
                    fr = op_frontiers[unique_id]

                    # pick operator with highest posterior mean quality
                    max_quality_op = max(
                        fr.get_frontier_ops(),
                        key=lambda op: fr.posteriors[op.get_full_op_id()]["quality"].mean
                    )

                    # get inputs
                    inputs = fr.get_frontier_op_inputs(
                        source_indices_to_sample,
                        max_quality_op
                    )
                    inputs = [x for x in inputs if x[-1] is not None]
                    if not inputs:
                        break

                    # execute ops and update samples
                    try:
                        out, num_ops = self._execute_op_set(unique_id, inputs)
                        samples_drawn += num_ops
                    except Exception as e:
                        logger.error(f"Error executing op set for {unique_id}: {e}")
                        # Skip this iteration and continue
                        break

                    # score outputs
                    all_record_sets = {
                        idx: [(rs, op) for rs, op, _ in tuples]
                        for idx, tuples in out.items()
                    }
                    all_record_sets, val_stats = self._score_quality(
                        validator, all_record_sets
                    )

                    # update plan stats
                    new_stats = []
                    for tuples in out.values():
                        for rs, _, is_new in tuples:
                            if is_new:
                                new_stats.extend(rs.record_op_stats)

                    plan_stats.add_record_op_stats(unique_id, new_stats)
                    plan_stats.add_validation_gen_stats(unique_id, val_stats)

                    # feed into next op
                    next_id = plan.get_next_unique_logical_op_id(unique_id)
                    if next_id:
                        best_sets = {
                            idx: [rs for rs, _ in record_list]
                            for idx, record_list in all_record_sets.items()
                        }
                        op_frontiers[next_id].update_inputs(unique_id, best_sets)

                    # update Bayesian frontier
                    fr.update_frontier(unique_id, plan_stats)
                
                # Check if we made progress in this iteration
                if samples_drawn == samples_before_iteration:
                    consecutive_no_progress_iterations += 1
                    logger.warning(f"No progress in iteration (count: {consecutive_no_progress_iterations}/{max_no_progress_iterations})")
                    if consecutive_no_progress_iterations >= max_no_progress_iterations:
                        logger.error(f"Stopping execution: No progress for {max_no_progress_iterations} consecutive iterations")
                        break
                else:
                    consecutive_no_progress_iterations = 0

        finally:
            self.progress_manager.finish()

        plan_stats.finish()
        return plan_stats