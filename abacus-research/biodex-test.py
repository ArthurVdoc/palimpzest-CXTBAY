import argparse
import json
import os
import time

import chromadb
import datasets
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.policy import MaxQualityAtFixedCost

from datetime import datetime

import httpx
import litellm

deepinfra_cost = {
    "input_cost_per_token": 4e-05,
    "output_cost_per_token": 1e-04,
    "max_tokens": 20000
}

litellm.model_cost["meta-llama/Llama-3.1-8B-Instruct"] = deepinfra_cost

biodex_entry_cols = [
    {"name": "pmid", "type": str, "desc": "The PubMed ID of the medical paper"},
    {"name": "title", "type": str, "desc": "The title of the medical paper"},
    {"name": "abstract", "type": str, "desc": "The abstract of the medical paper"},
    {"name": "fulltext", "type": str, "desc": "The full text of the medical paper, which contains information relevant for creating a drug safety report."},
]

biodex_reactions_cols = [
    {"name": "reactions",
     "type": list[str],
     "desc": (
            "Extract medical conditions from the text. "
            "Example: Input: 'Patient has flu.' -> Output: {\"reactions\": [\"flu\"]}. "
            "SYSTEM INSTRUCTIONS: "
            "1. Output ONLY the valid JSON object. "
            "2. Do NOT repeat the input text. "
            "3. Do NOT output reasoning or notes. "
            "4. Use strictly DOUBLE QUOTES (\"). "
            "5. Stop immediately after closing brace }."
        ),
    }
]

biodex_reaction_labels_cols = [
    {"name": "reaction_labels",
     "type": list[str],
     "desc": (
            "Standardize the medical conditions. "
            "Example: Input: 'flu' -> Output: {\"reaction_labels\": [\"influenza\"]}. "
            "SYSTEM INSTRUCTIONS: "
            "1. Output ONLY the valid JSON object. "
            "2. Key MUST be \"reaction_labels\". "
            "3. Do NOT repeat the input text. "
            "4. Do NOT output markdown (```json). "
            "5. Stop immediately after closing brace }."
        ),
    },
]

biodex_ranked_reactions_labels_cols = [
    {"name": "ranked_reaction_labels",
     "type": list[str],
     "desc": (
            "Rank conditions by relevance to the patient's main complaint. "
            "SYSTEM INSTRUCTIONS: "
            "1. Output ONLY the valid JSON object. "
            "2. Key MUST be \"ranked_reaction_labels\". "
            "3. Do NOT output reasoning, explanations, or 'Chain of Thought'. "
            "4. Do NOT repeat the input. "
            "5. Format example: {\"ranked_reaction_labels\": [\"primary_condition\", \"secondary_condition\"]}"
        ),
    },
]


class BiodexValidator(pz.Validator):
    def __init__(
        self,
        rp_at_k: int = 5,
        num_samples: int = 5,
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__()

        dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions", split="train").to_pandas()
        if shuffle:
            dataset = dataset.sample(n=num_samples, random_state=seed).to_dict(orient="records")
        else:
            dataset = dataset.to_dict(orient="records")[:num_samples]

        self.pmid_to_label = self._compute_pmid_to_label(dataset)
        self.k = rp_at_k

    def _compute_pmid_to_label(self, dataset: list[dict]) -> dict:
        pmid_to_label = {}
        for entry in dataset:
            pmid = str(entry["pmid"])
            reactions_lst = [
                reaction.strip().lower().replace("'", "").replace("^", "")
                for reaction in entry["reactions"].split(",")
            ]
            pmid_to_label[pmid] = reactions_lst
        return pmid_to_label

    def rank_precision_at_k(self, preds: list | None, targets: list):
        if preds is None:
            return 0.0
        try:
            preds = [pred.strip().lower().replace("'", "").replace("^", "") for pred in preds]
            targets = set([target.strip().lower().replace("'", "").replace("^", "") for target in targets])
            rn = len(targets)
            denom = min(self.k, rn)
            total = 0.0
            for i in range(self.k):
                total += preds[i] in targets if i < len(preds) else 0.0
            return total / denom
        except Exception:
            os.makedirs("rp@k-errors", exist_ok=True)
            ts = time.time()
            with open(f"rp@k-errors/error-{ts}.txt", "w") as f:
                f.write(str(preds))
            return 0.0

    def term_recall(self, preds: list | None, targets: list):
        if preds is None:
            return 0.0
        try:
            pred_terms = set([
                term.strip()
                for pred in preds
                for term in pred.lower().replace("'", "").replace("^", "").split(" ")
            ])
            target_terms = ([
                term.strip()
                for target in targets
                for term in target.lower().replace("'", "").replace("^", "").split(" ")
            ])
            intersect = pred_terms.intersection(target_terms)
            term_recall = len(intersect) / len(target_terms)
            return term_recall
        except Exception:
            os.makedirs("term-recall-eval-errors", exist_ok=True)
            ts = time.time()
            with open(f"term-recall-eval-errors/error-{ts}.txt", "w") as f:
                f.write(str(preds))
            return 0.0

    def map_score_fn(self, fields: list[str], input_record: dict, output: dict) -> float | None:
        field_name = fields[0]
        if field_name == "reactions":
            preds = output.get(field_name)
            targets = self.pmid_to_label[str(input_record["pmid"])]
            return self.term_recall(preds, targets)
        elif field_name == "ranked_reaction_labels":
            preds = output.get(field_name)
            targets = self.pmid_to_label[str(input_record["pmid"])]
            return self.rank_precision_at_k(preds, targets)
        else:
            raise NotImplementedError(f"Validator.map_score_fn not implemented for field {field_name}.")

    def retrieve_score_fn(self, fields: list[str], input_record: dict, output: dict) -> float | None:
        field_name = fields[0]
        if field_name == "reaction_labels":
            preds = output.get(field_name)
            targets = self.pmid_to_label[input_record["pmid"]]
            return self.term_recall(preds, targets)
        else:
            raise NotImplementedError(f"Validator.retrieve_score_fn not implemented for field {field_name}.")


class BiodexDataset(pz.IterDataset):
    def __init__(
        self,
        rp_at_k: int = 5,
        num_samples: int = 5,
        split: str = "test",
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(id="biodex", schema=biodex_entry_cols)

        self.dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions", split=split).to_pandas()
        if shuffle:
            self.dataset = self.dataset.sample(n=num_samples, random_state=seed).to_dict(orient="records")
        else:
            self.dataset = self.dataset.to_dict(orient="records")[:num_samples]

        self.rp_at_k = rp_at_k
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        entry = self.dataset[idx]
        pmid = entry["pmid"]
        title = entry["title"]
        abstract = entry["abstract"]
        fulltext = entry["fulltext"]
        item = {"pmid": pmid, "title": title, "abstract": abstract, "fulltext": fulltext}
        return item


if __name__ == "__main__":
    def log(msg: str):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    parser = argparse.ArgumentParser(description="Teste do pareto com MaxQualityAtFixedCost")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--progress", default=False, action="store_true")
    parser.add_argument(
        "--execution-strategy",
        default="parallel",
        type=str,
    )
    parser.add_argument(
        "--sentinel-execution-strategy",
        default="mab",
        type=str,
    )
    parser.add_argument(
        "--optimizer-strategy",
        default="pareto",
        type=str,
        help="pareto ou greedy",
    )
    parser.add_argument("--val-examples", default=25, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--j", default=3, type=int)
    parser.add_argument("--sample-budget", default=20, type=int)
    parser.add_argument(
        "--cost",
        default=1.0,
        type=float,
        help="Custo maximo (max_cost) para a politica MaxQualityAtFixedCost",
    )
    parser.add_argument("--exp-name", default=None, type=str)
    parser.add_argument("--priors-file", default=None, type=str)

    args = parser.parse_args()

    os.makedirs("opt-profiling-data", exist_ok=True)

    verbose = args.verbose
    progress = args.progress
    seed = args.seed
    val_examples = args.val_examples
    k = args.k
    j = args.j
    sample_budget = args.sample_budget
    execution_strategy = args.execution_strategy
    sentinel_execution_strategy = args.sentinel_execution_strategy
    optimizer_strategy = args.optimizer_strategy
    cost = args.cost
    exp_name = (
        f"biodex-pareto-test-{optimizer_strategy}-k{k}-j{j}-budget{sample_budget}-cost{cost}-seed{seed}"
        if args.exp_name is None
        else args.exp_name
    )
    priors = None
    if args.priors_file is not None:
        with open(args.priors_file) as f:
            priors = json.load(f)

    os.makedirs("tests", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"biodex_{timestamp}_strategy-{optimizer_strategy}_budget-{sample_budget}.log"
    log_path = os.path.join("tests", log_filename)

    # HIPOTESE SENDO TESTADA: pareto precisa de uma politica com dois eixos
    # (custo E qualidade) para ter uma fronteira nao-vazia, diferente de
    # MaxQuality() puro, que so tem um eixo.
    policy = pz.MaxQuality()
    #policy = MaxQualityAtFixedCost(max_cost=cost)
    print(f"USING POLICY: {policy}")

    validator = BiodexValidator(
        rp_at_k=5,
        num_samples=val_examples,
        shuffle=True,
        seed=seed,
    )

    train_dataset = BiodexDataset(
        split="train",
        num_samples=val_examples,
        shuffle=True,
        seed=seed,
    )
    train_dataset = {train_dataset.id: train_dataset}

    # embedding local, direto no vLLM de embeddings (porta 8002)
    chroma_client = chromadb.PersistentClient(".chroma-biodex")
    openai_ef = OpenAIEmbeddingFunction(
        api_key="EMPTY",
        model_name="nomic-ai/nomic-embed-text-v1",
        api_base="http://localhost:8002/v1",
    )
    index = chroma_client.get_collection("biodex-reaction-terms", embedding_function=openai_ef)

    def search_func(index: chromadb.Collection, query: list[list[float]], k: int) -> list[str]:
        results = index.query(query, n_results=5)
        final_results = []
        for query_docs, query_distances in zip(results["documents"], results["distances"]):
            for doc, dist in zip(query_docs, query_distances):
                cosine_similarity = 1 - dist
                final_results.append({"content": doc, "similarity": cosine_similarity})
        sorted_results = sorted(final_results, key=lambda result: result["similarity"], reverse=True)
        sorted_results_set = set()
        final_sorted_results = []
        for result in sorted_results:
            if result["content"] not in sorted_results_set:
                sorted_results_set.add(result["content"])
                final_sorted_results.append(result["content"])
        return {"reaction_labels": final_sorted_results[:k]}

    # construct plan
    plan = BiodexDataset(split="test", num_samples=10, shuffle=True, seed=seed)
    plan = plan.sem_map(biodex_reactions_cols)
    plan = plan.retrieve(
        index=index,
        search_func=search_func,
        search_attr="reactions",
        output_attrs=biodex_reaction_labels_cols,
    )
    plan = plan.sem_map(biodex_ranked_reactions_labels_cols, depends_on=["title", "abstract", "fulltext", "reaction_labels"])

    # execute pz plan, chamadas de LLM passam pelo scheduler local (porta 9000)
    config = pz.QueryProcessorConfig(
        available_models=[
            "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
        ],
        api_base="http://localhost:9000/v1",
        policy=policy,
        optimizer_strategy=optimizer_strategy,
        sentinel_execution_strategy=sentinel_execution_strategy,
        execution_strategy=execution_strategy,
        use_final_op_quality=False,
        max_workers=64,
        verbose=verbose,
        allow_bonded_query=True,
        allow_critic=True,
        allow_mixtures=True,
        allow_rag_reduction=True,
        progress=progress,
        k=k,
        j=j,
        sample_budget=sample_budget,
        seed=seed,
        exp_name=exp_name,
        priors=priors,
        tie_percentile=0.25,
    )

    data_record_collection = plan.optimize_and_run(config=config, train_dataset=train_dataset, validator=validator)

    print(data_record_collection.to_df())
    data_record_collection.to_df().to_csv(f"opt-profiling-data/{exp_name}-output.csv", index=False)

    records_path = f"opt-profiling-data/{exp_name}-records.json"
    stats_path = f"opt-profiling-data/{exp_name}-profiling.json"

    record_jsons = []
    for record in data_record_collection:
        record_dict = record.to_dict()
        record_dict = {
            k: v
            for k, v in record_dict.items()
            if k in ["pmid", "reactions", "reaction_labels", "ranked_reaction_labels"]
        }
        record_jsons.append(record_dict)

    with open(records_path, "w") as f:
        json.dump(record_jsons, f)

    execution_stats_dict = data_record_collection.execution_stats.to_json()
    with open(stats_path, "w") as f:
        json.dump(execution_stats_dict, f)

    test_dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions", split="test").to_pandas()
    test_dataset = test_dataset.sample(n=250, random_state=seed).to_dict(orient="records")

    def compute_target_record(entry):
        reactions_lst = [
            reaction.strip().lower().replace("'", "").replace("^", "")
            for reaction in entry["reactions"].split(",")
        ]
        return {"ranked_reaction_labels": reactions_lst}

    label_fields_to_values = {
        entry["pmid"]: compute_target_record(entry) for entry in test_dataset
    }

    def rank_precision_at_k(preds: list, targets: list, k: int):
        if preds is None:
            return 0.0
        preds = [pred.lower().replace("'", "").replace("^", "") for pred in preds]
        targets = set([target.lower().replace("'", "").replace("^", "") for target in targets])
        rn = len(targets)
        denom = min(k, rn)
        total = 0.0
        for i in range(k):
            total += preds[i] in targets if i < len(preds) else 0.0
        return total / denom

    def compute_avg_rp_at_k(records, k=5):
        total_rp_at_k = 0
        bad = 0
        for record in records:
            pmid = record['pmid']
            preds = record['ranked_reaction_labels']
            targets = label_fields_to_values[pmid]['ranked_reaction_labels']
            try:
                total_rp_at_k += rank_precision_at_k(preds, targets, k)
            except Exception:
                bad += 1
        return total_rp_at_k / len(records), bad

    rp_at_k, bad = compute_avg_rp_at_k(record_jsons, k=5)
    final_plan_id = list(data_record_collection.execution_stats.plan_stats.keys())[0]
    final_plan_str = data_record_collection.execution_stats.plan_strs[final_plan_id]
    stats_dict = {
        "rp@5": rp_at_k,
        "optimization_time": data_record_collection.execution_stats.optimization_time,
        "optimization_cost": data_record_collection.execution_stats.optimization_cost,
        "plan_execution_time": data_record_collection.execution_stats.plan_execution_time,
        "plan_execution_cost": data_record_collection.execution_stats.plan_execution_cost,
        "total_execution_time": data_record_collection.execution_stats.total_execution_time,
        "total_execution_cost": data_record_collection.execution_stats.total_execution_cost,
        "plan_str": final_plan_str,
    }
    with open(f"opt-profiling-data/{exp_name}-metrics.json", "w") as f:
        json.dump(stats_dict, f)

    log("========== EXPERIMENT RESULTS ==========")
    log(f"Experiment name: {exp_name}")
    log(f"Sentinel execution strategy: {sentinel_execution_strategy}")
    log(f"Execution strategy: {execution_strategy}")
    log(f"Optimizer strategy: {optimizer_strategy}")
    log(f"Sample budget: {sample_budget}")
    log(f"k: {k}")
    log(f"j: {j}")
    log(f"Seed: {seed}")
    log(f"Policy: {policy}")
    log("FINAL METRICS:")
    log(f"bad: {bad}")
    log(f"rp@5: {rp_at_k:.5f}")
    log(f"Optimization time: {data_record_collection.execution_stats.optimization_time}")
    log(f"Optimization cost: {data_record_collection.execution_stats.optimization_cost}")
    log(f"Plan execution time: {data_record_collection.execution_stats.plan_execution_time}")
    log(f"Plan execution cost: {data_record_collection.execution_stats.plan_execution_cost}")
    log(f"Total execution time: {data_record_collection.execution_stats.total_execution_time}")
    log(f"Total execution cost: {data_record_collection.execution_stats.total_execution_cost}")
    log("===================================")