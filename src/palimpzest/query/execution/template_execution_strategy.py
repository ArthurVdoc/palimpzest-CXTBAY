'''
template para a conversão de estratégias para o contexto do abacus.
com o MAB do abacus possui diversas particularidades, é preciso
adaptar estratégias diferentes para que possam ser utilizadas.
'''

import logging
import numpy as np

from palimpzest.core.data.dataset import Dataset
from palimpzest.core.elements.records import DataRecordSet
from palimpzest.core.models import SentinelPlanStats
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy import SentinelExecutionStrategy
from palimpzest.query.optimizer.plan import SentinelPlan
from palimpzest.validator.validator import Validator
from palimpzest.utils.progress import create_progress_manager

logger = logging.getLogger(__name__)

class TempOpFrontier:
    '''
    essa classe representa o conjunto de operadores que estão atualmente na fronteira para algum operador lógico.
    cada operador na fronteira é uma instância de PhysicalOperatorCandidate, que:
    1. está na fronteira de pareto do conjunto de operadores testados, ou
    2. foi testtado menos que j vezes.

    cada operador lógico possui:
    - frontier_ops  (operadores candidatos)
    - reservoir_ops (operadores restantes)
    - track de inputs recebidos do pai
    '''

    def __init__(
        self,
        op_set,
        source_unique_ids,
        root_dataset_ids,
        source_indices,
        k,
        j,
        seed,
        policy: Policy,
    ):
        self.k = min(k, len(op_set))
        self.j = j
        self.policy = policy
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.op_set = op_set
        self.frontier_ops = op_set[:self.k]
        self.reservoir_ops = op_set[self.k:]

        # Mapeia inputs vindos de operadores anteriores
        # {source_unique_logical_op_id: {source_idx: [inputs...]}}
        self.source_inputs = {src: {} for src in source_unique_ids}
        self.source_indices = source_indices  # lista de índices possíveis

    # operator selection
    def select_operator(self):
        '''
        retorna um operador físico da fronteira.
        aqui a estratégia é implementada.
        '''

        # exemplo trivial: sempre o primeiro
        return self.frontier_ops[0]

    # seleção de inputs
    def get_op_inputs(self):
        '''
        retorna uma lista de (operator, source_idx, inputs)
        aqui a lógica deve ser substituída.
        '''
        chosen_op = self.select_operator()
        inputs = []

        for src_unique_id, src_map in self.source_inputs.items():
            for source_idx, inp_list in src_map.items():
                for inp in inp_list:
                    if inp is not None:
                        inputs.append((chosen_op, source_idx, inp))

        return inputs

    # atualização da fronteira
    def update_frontier(self, unique_id, plan_stats):
        '''
        usa plan_stats para atualizar a fronteira de operadores.
        (pareto, lcb/ucb, heurística, etc)
        '''

        # exemplo trivial: não muda a fronteira
        pass


class TemplateExecutionStrategy(SentinelExecutionStrategy):
    '''
    estratégia de execução que usa TempOpFrontier para gerenciar operadores.
    '''

    def _execute_sentinel_plan(self, plan, frontiers, validator, plan_stats):
        samples_drawn = 0

        while samples_drawn < self.sample_budget:

            # percorre operadores em ordem topológica
            for topo_idx, (logical_op_id, _) in enumerate(plan):
                unique_id = f"{topo_idx}-{logical_op_id}"

                frontier = frontiers[unique_id]
                op_inputs = frontier.get_op_inputs()
                op_inputs = [t for t in op_inputs if t[-1] is not None]

                if len(op_inputs) == 0:
                    break

                # executa operadores físicos
                src_to_recordsets, num_llm = self._execute_op_set(unique_id, op_inputs)
                samples_drawn += num_llm

                # calcula qualidade
                scored, val_stats = self._score_quality(
                    validator,
                    {src: [(rs, op) for rs, op, _ in arr] for src, arr in src_to_recordsets.items()}
                )

                # registra métricas novas
                new_stats = []
                for _, tuples in src_to_recordsets.items():
                    for record_set, _, is_new in tuples:
                        if is_new:
                            new_stats.extend(record_set.record_op_stats)

                plan_stats.add_record_op_stats(unique_id, new_stats)
                plan_stats.add_validation_gen_stats(unique_id, val_stats)

                # propaga inputs ao próximo operador
                next_id = plan.get_next_unique_logical_op_id(unique_id)
                if next_id is not None:
                    cleaned = {src: [rs for rs, _ in lst] for src, lst in scored.items()}
                    frontiers[next_id].update_inputs(unique_id, cleaned)

                # atualiza fronteira do operador atual
                frontier.update_frontier(unique_id, plan_stats)

        plan_stats.finish()
        return plan_stats

    def execute_sentinel_plan(self, plan, train_dataset, validator):
        logger.info("Executing NEW Sentinel Plan")

        plan_stats = SentinelPlanStats.from_plan(plan)
        plan_stats.start()

        # prepara sampling dos datasets
        shuffled = {}
        for ds_id, ds in train_dataset.items():
            idxs = [f"{ds_id}-{i}" for i in range(len(ds))]
            self.rng.shuffle(idxs)
            shuffled[ds_id] = idxs

        # construção das fronteiras iniciais
        frontiers = {}

        for topo_idx, (logical_op_id, op_set) in enumerate(plan):
            unique_id = f"{topo_idx}-{logical_op_id}"

            source_ids = plan.get_source_unique_logical_op_ids(unique_id)
            root_ids = plan.get_root_dataset_ids(unique_id)
            sample_op = op_set[0]

            if sample_op.is_scan():
                ds_id = root_ids[0]
                source_indices = shuffled[ds_id]
            else:
                src_parent = frontiers[source_ids[0]]
                source_indices = src_parent.source_indices

            frontiers[unique_id] = NewOpFrontier(
                op_set, source_ids, root_ids, source_indices,
                self.k, self.j, self.seed, self.policy
            )

        # barra de progresso
        self.progress_manager = create_progress_manager(
            plan, self.sample_budget, self.progress
        )
        self.progress_manager.start()

        try:
            plan_stats = self._execute_sentinel_plan(plan, frontiers, validator, plan_stats)
        finally:
            self.progress_manager.finish()

        logger.info(f"Done executing NEW strategy for plan {plan.plan_id}")
        return plan_stats     

