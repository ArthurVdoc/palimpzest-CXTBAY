from __future__ import annotations

import logging
from copy import deepcopy

from pydantic.fields import FieldInfo

from palimpzest.constants import Model
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.lib.schemas import get_schema_field_names
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy_type import ExecutionStrategyType
from palimpzest.query.operators.logical import (
    ComputeOperator,
    ConvertScan,
    Distinct,
    FilteredScan,
    JoinOp,
    LimitScan,
    Project,
    SearchOperator,
)
from palimpzest.query.optimizer import (
    IMPLEMENTATION_RULES,
    TRANSFORMATION_RULES,
)
from palimpzest.query.optimizer.cost_model import BaseCostModel, SampleBasedCostModel
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.query.optimizer.primitives import Group, LogicalExpression
from palimpzest.query.optimizer.rules import (
    CriticAndRefineConvertRule,
    LLMConvertBondedRule,
    MixtureOfAgentsConvertRule,
    RAGConvertRule,
    SplitConvertRule,
)
from palimpzest.query.optimizer.tasks import (
    ApplyRule,
    ExpandGroup,
    OptimizeGroup,
    OptimizeLogicalExpression,
    OptimizePhysicalExpression,
)

logger = logging.getLogger(__name__)


class Optimizer:
    """
    The optimizer is responsible for searching the space of possible physical plans
    for a user's initial (logical) plan and selecting the one which is closest to
    optimizing the user's policy objective.

    This optimizer is modeled after the Cascades framework for top-down query optimization:
    - Thesis describing Cascades implementation (Chapters 1-3):
      https://15721.courses.cs.cmu.edu/spring2023/papers/17-optimizer2/xu-columbia-thesis1998.pdf

    - Andy Pavlo lecture with walkthrough example: https://www.youtube.com/watch?v=PXS49-tFLcI

    - Original Paper: https://www.cse.iitb.ac.in/infolab/Data/Courses/CS632/2015/Papers/Cascades-graefe.pdf

    Notably, this optimization framework has served as the backbone of Microsoft SQL Server, CockroachDB,
    and a few other important DBMS systems.

    NOTE: the optimizer currently assumes that field names are unique across schemas; we do try to enforce
          this by rewriting field names underneath-the-hood to be "{schema_name}.{field_name}", but this still
          does not solve a situation in which -- for example -- a user uses the pz.URL schema twice in the same
          program. In order to address that situation, we will need to augment our renaming scheme.
    """
  #-------------------REVISED UP-------------------

  #------------------REVISED down------------------


    def __init__(
        self,
        policy: Policy,
        cost_model: BaseCostModel,
        available_models: list[Model],
        join_parallelism: int = 64,
        reasoning_effort: str | None = None,
        api_base: str | None = None,
        verbose: bool = False,
        allow_bonded_query: bool = True,
        allow_rag_reduction: bool = False,
        allow_mixtures: bool = True,
        allow_critic: bool = False,
        allow_split_merge: bool = False,
        optimizer_strategy: OptimizationStrategyType = OptimizationStrategyType.PARETO,
        execution_strategy: ExecutionStrategyType = ExecutionStrategyType.PARALLEL,
        use_final_op_quality: bool = False, # TODO: make this func(plan) -> final_quality
        **kwargs,
    ):
        print(f"[INIT] COST MODEL PARAMS: {cost_model}")
        # store the policy
        self.policy = policy

        # store the cost model
        self.cost_model = cost_model

        # mapping from each group id to its Group object
        self.groups = {}

        # mapping from each expression to its Expression object
        self.expressions = {}

        # the stack of tasks to perform during optimization
        self.tasks_stack = []

        # the lists of implementation and transformation rules that the optimizer can apply
        self.implementation_rules = IMPLEMENTATION_RULES
        self.transformation_rules = TRANSFORMATION_RULES

        # get the strategy class associated with the optimizer strategy
        optimizer_strategy_cls = optimizer_strategy.value
        self.strategy = optimizer_strategy_cls()

        # remove transformation rules for optimization strategies which do not require them
        if optimizer_strategy.no_transformation():
            self.transformation_rules = []

        # if we are not performing optimization, set available models to be single model
        # and remove all optimizations (except for bonded queries)
        if optimizer_strategy == OptimizationStrategyType.NONE:
            self.allow_bonded_query = True
            self.allow_rag_reduction = False
            self.allow_mixtures = False
            self.allow_critic = False
            self.allow_split_merge = False
            self.available_models = [available_models[0]]

        # store optimization hyperparameters
        self.verbose = verbose
        self.available_models = available_models
        self.join_parallelism = join_parallelism
        self.reasoning_effort = reasoning_effort
        self.api_base = api_base
        self.allow_bonded_query = allow_bonded_query
        self.allow_rag_reduction = allow_rag_reduction
        self.allow_mixtures = allow_mixtures
        self.allow_critic = allow_critic
        self.allow_split_merge = allow_split_merge
        self.optimizer_strategy = optimizer_strategy
        self.execution_strategy = execution_strategy
        self.use_final_op_quality = use_final_op_quality

        # prune implementation rules based on boolean flags
        if not self.allow_bonded_query:
            self.implementation_rules = [
                rule
                for rule in self.implementation_rules
                if rule not in [LLMConvertBondedRule]
            ]

        if not self.allow_rag_reduction:
            self.implementation_rules = [
                rule for rule in self.implementation_rules if not issubclass(rule, RAGConvertRule)
            ]

        if not self.allow_mixtures:
            self.implementation_rules = [
                rule for rule in self.implementation_rules if not issubclass(rule, MixtureOfAgentsConvertRule)
            ]

        if not self.allow_critic:
            self.implementation_rules = [
                rule for rule in self.implementation_rules if not issubclass(rule, CriticAndRefineConvertRule)
            ]

        if not self.allow_split_merge:
            self.implementation_rules = [
                rule for rule in self.implementation_rules if not issubclass(rule, SplitConvertRule)
            ]

        logger.info(f"Initialized Optimizer with verbose={self.verbose}")
        logger.debug(f"Initialized Optimizer with params: {self.__dict__}")

  #-------------------REVISED UP-------------------
        self._captured_full_space = False
        self._optimize_call_count = 0
  #------------------REVISED down------------------

    def update_cost_model(self, cost_model: BaseCostModel):
        self.cost_model = cost_model

    def get_physical_op_params(self):
        return {
            "verbose": self.verbose,
            "available_models": self.available_models,
            "join_parallelism": self.join_parallelism,
            "reasoning_effort": self.reasoning_effort,
            "api_base": self.api_base,          
        }

    def deepcopy_clean(self):
        optimizer = Optimizer(
            policy=self.policy,
            cost_model=SampleBasedCostModel(),
            verbose=self.verbose,
            available_models=self.available_models,
            join_parallelism=self.join_parallelism,
            reasoning_effort=self.reasoning_effort,
            api_base=self.api_base,
            allow_bonded_query=self.allow_bonded_query,
            allow_rag_reduction=self.allow_rag_reduction,
            allow_mixtures=self.allow_mixtures,
            allow_critic=self.allow_critic,
            allow_split_merge=self.allow_split_merge,
            optimizer_strategy=self.optimizer_strategy,
            execution_strategy=self.execution_strategy,
            use_final_op_quality=self.use_final_op_quality,
        )
        return optimizer

    def update_strategy(self, optimizer_strategy: OptimizationStrategyType):
        self.optimizer_strategy = optimizer_strategy
        optimizer_strategy_cls = optimizer_strategy.value
        self.strategy = optimizer_strategy_cls()

    def construct_group_tree(self, dataset: Dataset) -> tuple[int, dict[str, FieldInfo], dict[str, set[str]]]:
        logger.debug(f"Constructing group tree for dataset: {dataset}")
        ### convert node --> Group ###
        # create the op for the given node
        op = dataset._operator

        # compute the input group id(s) and field(s) for this node
        if len(dataset._sources) == 0:
            input_group_ids, input_group_fields, input_group_properties = ([], {}, {})
        elif len(dataset._sources) == 1:
            input_group_id, input_group_fields, input_group_properties = self.construct_group_tree(dataset._sources[0])
            input_group_ids = [input_group_id]
        elif len(dataset._sources) == 2:
            left_input_group_id, left_input_group_fields, left_input_group_properties = self.construct_group_tree(dataset._sources[0])
            right_input_group_id, right_input_group_fields, right_input_group_properties = self.construct_group_tree(dataset._sources[1])
            input_group_ids = [left_input_group_id, right_input_group_id]
            input_group_fields = {**left_input_group_fields, **right_input_group_fields}
            input_group_properties = deepcopy(left_input_group_properties)
            for k, v in right_input_group_properties.items():
                if k in input_group_properties:
                    input_group_properties[k].update(v)
                else:
                    input_group_properties[k] = deepcopy(v)
        else:
            raise NotImplementedError("Constructing group trees for datasets with more than 2 sources is not supported.")

        # compute the fields added by this operation and all fields
        input_group_short_field_names = list(
            map(lambda full_field: full_field.split(".")[-1], input_group_fields.keys())
        )
        new_fields = {
            field_name: op.output_schema.model_fields[field_name.split(".")[-1]]
            for field_name in get_schema_field_names(op.output_schema, id=dataset.id)
            if (field_name not in input_group_short_field_names) or (hasattr(op, "udf") and op.udf is not None)
        }
        all_fields = {**input_group_fields, **new_fields}

        # compute the set of (short) field names this operation depends on
        depends_on_field_names = (
            {} if dataset.is_root else {field_name.split(".")[-1] for field_name in op.depends_on}
        )

        # NOTE: group_id is computed as the unique (sorted) set of fields and properties;
        #       If an operation does not modify the fields (or modifies them in a way that
        #       can create an idential field set to an earlier group) then we must add an
        #       id from the operator to disambiguate the two groups.
        # compute all properties including this operations'
        all_properties = deepcopy(input_group_properties)
        if isinstance(op, ConvertScan) and sorted(op.input_schema.model_fields.keys()) == sorted(op.output_schema.model_fields.keys()):
            model_fields_dict = {
                k: {"annotation": v.annotation, "default": v.default, "description": v.description}
                for k, v in op.output_schema.model_fields.items()
            }
            if "maps" in all_properties:
                all_properties["maps"].add(model_fields_dict)
            else:
                all_properties["maps"] = set([model_fields_dict])

        elif isinstance(op, FilteredScan):
            # NOTE: we could use op.get_full_op_id() here, but storing filter strings makes
            #       debugging a bit easier as you can read which filters are in the Group
            op_filter_str = op.filter.get_filter_str()
            if "filters" in all_properties:
                all_properties["filters"].add(op_filter_str)
            else:
                all_properties["filters"] = set([op_filter_str])

        elif isinstance(op, JoinOp):
            if "joins" in all_properties:
                all_properties["joins"].add(op.condition)
            else:
                all_properties["joins"] = set([op.condition])

        elif isinstance(op, LimitScan):
            op_limit_str = op.get_logical_op_id()
            if "limits" in all_properties:
                all_properties["limits"].add(op_limit_str)
            else:
                all_properties["limits"] = set([op_limit_str])

        elif isinstance(op, Project):
            op_project_str = op.get_logical_op_id()
            if "projects" in all_properties:
                all_properties["projects"].add(op_project_str)
            else:
                all_properties["projects"] = set([op_project_str])

        elif isinstance(op, Distinct):
            op_distinct_str = op.get_logical_op_id()
            if "distincts" in all_properties:
                all_properties["distincts"].add(op_distinct_str)
            else:
                all_properties["distincts"] = set([op_distinct_str])

        # TODO: temporary fix; perhaps use op_ids to identify group?
        elif isinstance(op, ComputeOperator):
            op_instruction = op.instruction
            if "instructions" in all_properties:
                all_properties["instructions"].add(op_instruction)
            else:
                all_properties["instructions"] = set([op_instruction])

        elif isinstance(op, SearchOperator):
            op_search_query = op.search_query
            if "search_queries" in all_properties:
                all_properties["search_queries"].add(op_search_query)
            else:
                all_properties["search_queries"] = set([op_search_query])

        # construct the logical expression and group
        logical_expression = LogicalExpression(
            operator=op,
            input_group_ids=input_group_ids,
            input_fields=input_group_fields,
            depends_on_field_names=depends_on_field_names,
            generated_fields=new_fields,
            group_id=None,
        )
        group = Group(
            logical_expressions=[logical_expression],
            fields=all_fields,
            properties=all_properties,
        )
        logical_expression.set_group_id(group.group_id)

        # add the expression and group to the optimizer's expressions and groups and return
        self.expressions[logical_expression.expr_id] = logical_expression
        self.groups[group.group_id] = group
        logger.debug(f"Constructed group tree for dataset: {dataset}")
        logger.debug(f"Group: {group.group_id}, {all_fields}, {all_properties}")

        return group.group_id, all_fields, all_properties

    def convert_query_plan_to_group_tree(self, dataset: Dataset) -> str:
        logger.debug(f"Converting query plan to group tree for dataset: {dataset}")

        # compute depends_on field for every node
        short_to_full_field_name = {}
        for node in dataset:
            # update mapping from short to full field names
            short_field_names = get_schema_field_names(node.schema)
            full_field_names = get_schema_field_names(node.schema, id=node.id)
            for short_field_name, full_field_name in zip(short_field_names, full_field_names):
                # set mapping automatically if this is a new field
                if short_field_name not in short_to_full_field_name or (hasattr(node._operator, "udf") and node._operator.udf is not None):
                    short_to_full_field_name[short_field_name] = full_field_name

            # if the node is a root Dataset, then skip
            if node.is_root:
                continue

            # If the node already has depends_on specified, then resolve each field name to a full (unique) field name
            if len(node._operator.depends_on) > 0:
                node._operator.depends_on = list(map(lambda field: short_to_full_field_name[field], node._operator.depends_on))
                continue

            # otherwise, make the node depend on all upstream nodes
            node._operator.depends_on = set()
            upstream_nodes = node.get_upstream_datasets()
            for upstream_node in upstream_nodes:
                upstream_field_names = get_schema_field_names(upstream_node.schema, id=upstream_node.id)
                node._operator.depends_on.update(upstream_field_names)
            node._operator.depends_on = list(node._operator.depends_on)

        # construct tree of groups
        final_group_id, _, _ = self.construct_group_tree(dataset)

        logger.debug(f"Converted query plan to group tree for dataset: {dataset}")
        logger.debug(f"Final group id: {final_group_id}")

        return final_group_id

    def heuristic_optimization(self, group_id: int) -> None:
        """
        Apply universally desirable transformations (e.g. filter/projection push-down).
        """
        pass
    
#------------------REVISED UP------------------
    # def dump_all_physical_plans(self, base_filename="search_space.json"):
    def dump_all_physical_plans(self, base_filename="search_space.json", output_dir="search-spaces"):
        """
        Captura TODO o espaço de busca de operadores físicos.
        Gera um nome de arquivo único baseado em timestamp a partir do nome base.
        Exemplo: "search_space.json" -> "search_space_20250219_153045_123.json"

        Args:
            base_filename: nome base do arquivo (padrão "search_space.json")
        """
        import json
        from datetime import datetime
        import os

        #-------------------REVISED UP-------------------
        # Garante que o diretório dedicado exista
        os.makedirs(output_dir, exist_ok=True)
        #------------------REVISED down------------------

        # Gera timestamp único (ano-mês-dia_hora-minuto-segundo-milissegundo)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # até milissegundos

        # Separa nome base e extensão
        base, ext = os.path.splitext(base_filename)
        unique_filename = f"{base}_{timestamp}{ext}"

        #-------------------REVISED UP-------------------
        # monta o caminho completo
        full_filepath = os.path.join(output_dir, unique_filename)
        #------------------REVISED down------------------

        search_space = {
            "timestamp": datetime.now().isoformat(),
            "optimize_call_count": self._optimize_call_count,
            "total_groups": len(self.groups),
            "total_operators": 0,
            "groups": {}
        }
        
        for group_id, group in self.groups.items():
            if not group.physical_expressions:
                continue
                
            group_data = {
                "group_id": str(group_id),
                "num_operators": len(group.physical_expressions),
                "fields": {k: str(v.annotation) for k, v in group.fields.items()},
                "properties": {k: list(v) if isinstance(v, set) else str(v) for k, v in group.properties.items()},
                "operators": []
            }
            
            for i, expr in enumerate(group.physical_expressions):
                op = expr.operator
                
                # Dados básicos do operador
                op_data = {
                    "index": i,
                    "operator_class": op.__class__.__name__,
                    "full_op_id": op.get_full_op_id(),
                    "logical_op_id": op.logical_op_id,
                    "parameters": {},
                }
                
                # Extrai TODOS os parâmetros via get_id_params()
                try:
                    params = op.get_id_params()
                    # Converte tudo para string para serialização JSON
                    op_data["parameters"] = {k: str(v) for k, v in params.items()}
                except Exception as e:
                    op_data["parameters"] = {"error": str(e)}
                
                # Adiciona campos específicos importantes
                if hasattr(op, 'model'):
                    op_data['model'] = str(op.model) if op.model else None
                    
                if hasattr(op, 'proposer_models'):
                    op_data['proposer_models'] = [str(m) for m in op.proposer_models]
                    op_data['aggregator_model'] = str(op.aggregator_model) if hasattr(op, 'aggregator_model') else None
                    op_data['temperatures'] = list(op.temperatures) if hasattr(op, 'temperatures') else None
                    
                if hasattr(op, 'num_chunks_per_field'):
                    op_data['rag_params'] = {
                        'num_chunks_per_field': op.num_chunks_per_field,
                        'chunk_size': op.chunk_size if hasattr(op, 'chunk_size') else None
                    }
                    
                if hasattr(op, 'critic_model'):
                    op_data['critic_refine'] = {
                        'critic_model': str(op.critic_model),
                        'refine_model': str(op.refine_model) if hasattr(op, 'refine_model') else None
                    }
                
                # Custos (se disponíveis)
                if expr.plan_cost:
                    op_data['plan_cost'] = {
                        'cost': float(expr.plan_cost.cost),
                        'quality': float(expr.plan_cost.quality),
                        'time': float(expr.plan_cost.time),
                    }
                
                group_data["operators"].append(op_data)
                search_space["total_operators"] += 1
            
            search_space["groups"][str(group_id)] = group_data
        
        # # Salva em arquivo
        # with open(unique_filename, 'w', encoding='utf-8') as f:
        #     json.dump(search_space, f, indent=2, ensure_ascii=False)

        #-------------------REVISED UP-------------------
        # Salva em arquivo no caminho completo
        with open(full_filepath, 'w', encoding='utf-8') as f:
            json.dump(search_space, f, indent=2, ensure_ascii=False)
        #------------------REVISED down------------------
        
        # # Imprime resumo
        # print(f"\n{'='*80}")
        # print(f"📊 ESPAÇO DE BUSCA CAPTURADO")
        # print(f"{'='*80}")
        # print(f"Arquivo: {unique_filename}")
        # print(f"Timestamp: {timestamp}")
        # print(f"Call count: {self._optimize_call_count}")
        # print(f"Total de Grupos: {len(search_space['groups'])}")
        # print(f"Total de Operadores Físicos: {search_space['total_operators']}")
        # print(f"\nDetalhamento por Grupo:")

        #-------------------REVISED UP-------------------
        # Imprime resumo
        print(f"\n{'='*80}")
        print(f"📊 ESPAÇO DE BUSCA CAPTURADO")
        print(f"{'='*80}")
        print(f"Arquivo: {full_filepath}")
        print(f"Timestamp: {timestamp}")
        print(f"Call count: {self._optimize_call_count}")
        print(f"Total de Grupos: {len(search_space['groups'])}")
        print(f"Total de Operadores Físicos: {search_space['total_operators']}")
        print(f"\nDetalhamento por Grupo:")
        #------------------REVISED down------------------
        
        for gid, gdata in search_space["groups"].items():
            print(f"\n  📁 Grupo {gid}: {gdata['num_operators']} operadores")
            
            # Conta por tipo
            op_types = {}
            for op in gdata["operators"]:
                op_class = op["operator_class"]
                op_types[op_class] = op_types.get(op_class, 0) + 1
            
            for op_type, count in sorted(op_types.items()):
                print(f"      • {op_type}: {count}")
                
            # Mostra alguns exemplos de parâmetros
            if gdata["operators"]:
                print(f"      Exemplos de variações:")
                for op in gdata["operators"][:3]:  # Primeiros 3
                    if 'model' in op and op['model']:
                        print(f"        - {op['operator_class']} com {op['model']}")
                    elif 'proposer_models' in op:
                        print(f"        - MoA com {len(op['proposer_models'])} proposers")
        
        print(f"{'='*80}\n")
        
        return search_space
#-----------------REVISED DOWN-----------------

    def search_optimization_space(self, group_id: int) -> None:
        logger.debug(f"Searching optimization space for group_id: {group_id}")

        # begin the search for an optimal plan with a task to optimize the final group
        initial_task = OptimizeGroup(group_id)
        self.tasks_stack.append(initial_task)
        
        # ------------------------ REVISED --------------------------
        # while len(self.tasks_stack) > 0:
        #     task = self.tasks_stack.pop(-1)
        #     new_tasks = []
            
        #     if isinstance(task, ApplyRule):
        #         params = self.get_physical_op_params()
        #         # Verifica se os modelos estão chegando na tarefa
        #         models = params.get("available_models", [])
        #         rule_name = task.rule.__class__.__name__ if hasattr(task, 'rule') else "UnknownRule"
                
        #         print(f"\n[TASK DEBUG] Aplicando: {rule_name}")
        #         print(f"    - Modelos na tarefa: {[m for m in models]}")
                
        #         context = {"costed_full_op_ids": self.cost_model.get_costed_full_op_ids()}
        #         new_tasks = task.perform(
        #             self.groups, self.expressions, context=context, **params,
        #         )
                
        #         # Verifica se a aplicação da regra gerou novos caminhos
        #         if not new_tasks:
        #             print(f"    ⚠️ A regra {rule_name} NÃO gerou novas tarefas/planos físicos.")
        #     elif isinstance(task, (OptimizeGroup, ExpandGroup)):
        #         new_tasks = task.perform(self.groups)
        #     elif isinstance(task, OptimizeLogicalExpression):
        #         new_tasks = task.perform(self.transformation_rules, self.implementation_rules)
        #     elif isinstance(task, OptimizePhysicalExpression):
        #         context = {"optimizer_strategy": self.optimizer_strategy, "execution_strategy": self.execution_strategy}
        #         new_tasks = task.perform(self.cost_model, self.groups, self.policy, context=context)
            
        #     self.tasks_stack.extend(new_tasks)

        # logger.debug(f"Done searching optimization space for group_id: {group_id}")
        # ------------------------ REVISED --------------------------
            
        # TODO: conditionally stop when X number of tasks have been executed to limit exhaustive search
        while len(self.tasks_stack) > 0:
            task = self.tasks_stack.pop(-1)
            new_tasks = []
            if isinstance(task, (OptimizeGroup, ExpandGroup)):
                new_tasks = task.perform(self.groups)
            elif isinstance(task, OptimizeLogicalExpression):
                new_tasks = task.perform(self.transformation_rules, self.implementation_rules)
            elif isinstance(task, ApplyRule):
                context = {"costed_full_op_ids": self.cost_model.get_costed_full_op_ids()}
                new_tasks = task.perform(
                    self.groups, self.expressions, context=context, **self.get_physical_op_params(),
                )
            elif isinstance(task, OptimizePhysicalExpression):
                context = {"optimizer_strategy": self.optimizer_strategy, "execution_strategy": self.execution_strategy}
                new_tasks = task.perform(self.cost_model, self.groups, self.policy, context=context)
                
            self.tasks_stack.extend(new_tasks)
            
            
# ##------------------REVISED UP------------------
#             elif isinstance(task, ApplyRule):
#                 rule_name = task.rule.__class__.__name__
#                 params = self.get_physical_op_params()
#                 context = {"costed_full_op_ids": self.cost_model.get_costed_full_op_ids()}
                
#                 new_tasks = task.perform(
#                     self.groups, self.expressions, context=context, **params,
#                 )
                
#                 # DEBUG: Monitora por que os planos físicos não estão sendo criados
#                 if "Convert" in rule_name or "LLM" in rule_name or "Mixture" in rule_name:
#                     print(f">>> [RULE DEBUG] Regra: {rule_name}")
#                     print(f"    - Modelos disponíveis: {params.get('available_models')}")
#                     print(f"    - Tarefas geradas: {len(new_tasks)}")
#                     if not new_tasks:
#                         print(f"    ⚠️ A regra {rule_name} não gerou implementações físicas.")
# #--------------- REVISED DOWN------------------

        logger.debug(f"Done searching optimization space for group_id: {group_id}")
        

    #------------------REVISED UP------------------

    # def optimize(self, dataset: Dataset) -> list[PhysicalPlan]:
    #     """
    #     The optimize function takes in an initial query plan and searches the space of
    #     logical and physical plans in order to cost and produce a (near) optimal physical plan.
    #     """
    #     logger.info(f"Optimizing query plan: {dataset}")
    #     # compute the initial group tree for the user plan
    #     dataset_copy = dataset.copy()
    #     final_group_id = self.convert_query_plan_to_group_tree(dataset_copy)

    #     # TODO
    #     # # do heuristic based pre-optimization
    #     # self.heuristic_optimization(final_group_id)

    #     # search the optimization space by applying logical and physical transformations to the initial group tree
    #     self.search_optimization_space(final_group_id)
    #     logger.info(f"Getting optimal plans for final group id: {final_group_id}")

    #     return self.strategy.get_optimal_plans(self.groups, final_group_id, self.policy, self.use_final_op_quality)

    #-----------------REVISED DOWN-----------------

    def optimize(self, dataset: Dataset) -> list[PhysicalPlan]:
        """
        The optimize function takes in an initial query plan and searches the space of
        logical and physical plans in order to cost and produce a (near) optimal physical plan.
        """

        import os

        logger.info(f"Optimizing query plan: {dataset}")

    # ================= REVISED =================
        # Define o diretório de saída para os arquivos de análise
        output_dir = "search-spaces"
        os.makedirs(output_dir, exist_ok=True)
    # ===========================================
        
        # Incrementa contador de chamadas
        self._optimize_call_count += 1
        is_first_call = (self._optimize_call_count == 1)
        
        # compute the initial group tree for the user plan
        dataset_copy = dataset.copy()
        final_group_id = self.convert_query_plan_to_group_tree(dataset_copy)
        
        # search the optimization space by applying logical and physical transformations
        self.search_optimization_space(final_group_id)
        
        # # 🆕 CAPTURA 1: Espaço COMPLETO (APENAS na primeira chamada, ANTES do profiling)
        # if is_first_call:
        #     print("\n" + "="*80)
        #     print("📸 CAPTURANDO ESPAÇO DE BUSCA COMPLETO (PRÉ-PROFILING)")
        #     print("="*80)
        #     self.dump_all_physical_plans("search_space.json")
        
    #------------------REVISED UP------------------
        # 🆕 CAPTURA 1: Espaço COMPLETO (APENAS na primeira chamada, ANTES do profiling)
        if is_first_call:
            print("\n" + "="*80)
            print("📸 CAPTURANDO ESPAÇO DE BUSCA COMPLETO (PRÉ-PROFILING)")
            print("="*80)
            self.dump_all_physical_plans("search_space.json", output_dir=output_dir)
    #-----------------REVISED DOWN-----------------
        
        logger.info(f"Getting optimal plans for final group id: {final_group_id}")
        
        # Execute profiling/sampling e selecione o plano ótimo
        optimal_plans = self.strategy.get_optimal_plans(
            self.groups, final_group_id, self.policy, self.use_final_op_quality
        )
        
        # output_path = "debug_plans.txt"

    #------------------REVISED UP------------------
        # Modificado para salvar dentro da pasta
        output_path = os.path.join(output_dir, "debug_plans.txt")
    #-----------------REVISED DOWN-----------------
        
        with open(output_path, "a") as f:
            f.write("=== DEBUG: PLANOS ÓTIMOS (PARETO FRONTIER) ===\n")
            f.write(f"Total de planos encontrados: {len(optimal_plans)}\n")
            f.write(f"ID do Grupo Final: {final_group_id}\n")
            f.write("-" * 50 + "\n")
            
            if not optimal_plans:
                f.write("AVISO: A lista de planos está VAZIA.\n")
                f.write("Isto indica que a estratégia de Pareto descartou todos os candidatos\n")
                f.write("ou que nenhum plano físico foi gerado para o Biodex.\n")
            else:
                for i, plan in enumerate(optimal_plans):
                    f.write(f"\nPLANO #{i+1}:\n")
                    f.write(f"Estrutura: {plan}\n")
                    f.write(f"Custo Estimado: {getattr(plan, 'cost', 'N/D')}\n")
                    f.write(f"Qualidade Estimada: {getattr(plan, 'quality', 'N/D')}\n")
                    f.write("-" * 30 + "\n")
            
            f.write("\n=== FIM DO RELATÓRIO ===\n")

        print(f"✅ Debug dos planos guardado com sucesso em: {output_path}")
            
        # 🆕 CAPTURA 2: Espaço AMOSTRADO (APENAS na segunda+ chamadas, DEPOIS do profiling)
        if not is_first_call:
            print("\n" + "="*80)
            print("📸 CAPTURANDO ESPAÇO DE BUSCA AMOSTRADO (PÓS-PROFILING)")
            print("="*80)
            # self.dump_all_physical_plans("search_space.json")
            self.dump_all_physical_plans("search_space.json", output_dir=output_dir)
        
        return optimal_plans