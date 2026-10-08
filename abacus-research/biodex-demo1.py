import argparse
# =============== NEW FEATURE ===============
import sys
# =============== END NEW FEATURE ===============
import json
import os
import time

import chromadb
import datasets
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
import palimpzest as pz
from palimpzest.constants import Model

from datetime import datetime

import httpx
import litellm
deepinfra_cost = {
    "input_cost_per_token": 4e-05,
    "output_cost_per_token": 1e-04,
    "max_tokens": 20000
}

litellm.model_cost["meta-llama/Llama-3.1-8B-Instruct"] = deepinfra_cost
#litellm.model_cost["hosted_vllm/Qwen/Qwen2.5-7B-Instruct"] = deepinfra_cost

biodex_entry_cols = [
    {"name": "pmid", "type": str, "desc": "The PubMed ID of the medical paper"},
    {"name": "title", "type": str, "desc": "The title of the medical paper"},
    {"name": "abstract", "type": str, "desc": "The abstract of the medical paper"},
    {"name": "fulltext", "type": str, "desc": "The full text of the medical paper, which contains information relevant for creating a drug safety report."},
]

biodex_reactions_cols = [
    {"name": "reactions", 
     "type": list[str], 
    #  "desc": "The list of all medical conditions experienced by the patient as discussed in the report. Try to provide as many relevant medical conditions as possible."},
    "desc": (
            "Extract medical conditions from the text. "
            "Example: Input: 'Patient has flu.' -> Output: {\"reactions\": [\"flu\"]}. "
            # --- BLOCO DE SEGURANÇA ---
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
    #  "desc": "Official terms for medical conditions listed in `reactions`"},
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
    #  "desc": "The ranked list of medical conditions experienced by the patient. The most relevant label occurs first in the list. Be sure to rank ALL of the inputs."},
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

        # read dataset and prepare entries
        dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions", split="train").to_pandas()
        if shuffle:
            dataset = dataset.sample(n=num_samples, random_state=seed).to_dict(orient="records")
        else:
            dataset = dataset.to_dict(orient="records")[:num_samples]

        # compute mapping from pmid --> label (i.e. reactions list)
        self.pmid_to_label = self._compute_pmid_to_label(dataset)

        # store rp_at_k for computing rank-precision at k metric
        self.k = rp_at_k

    def _compute_pmid_to_label(self, dataset: list[dict]) -> dict:
        """Compute the label for a BioDEX report given its entry in the dataset."""
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
            # lower-case each list
            preds = [pred.strip().lower().replace("'", "").replace("^", "") for pred in preds]
            targets = set([target.strip().lower().replace("'", "").replace("^", "") for target in targets])

            # compute rank-precision at k
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
            # normalize terms in each list
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

            # compute term recall and return
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
            # =============== NEW FEATURE ===============
            # Depuracao: consistente com map_score_fn (evita KeyError se pmid vier como int).
            targets = self.pmid_to_label[str(input_record["pmid"])]
            # (original)
            # targets = self.pmid_to_label[input_record["pmid"]]
            # =============== END NEW FEATURE ===============
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
        # get entry
        entry = self.dataset[idx]

        # get input fields
        pmid = entry["pmid"]
        title = entry["title"]
        abstract = entry["abstract"]
        fulltext = entry["fulltext"]

        # create item with fields
        item = {"pmid": pmid, "title": title, "abstract": abstract, "fulltext": fulltext}

        return item


if __name__ == "__main__":
    def log(msg: str):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--progress", default=False, action="store_true", help="Print progress output")
    parser.add_argument("--constrained", default=False, action="store_true", help="Use constrained objective")
    parser.add_argument("--gpt4-mini-only", default=False, action="store_true", help="Use only GPT-4o-mini")
    parser.add_argument(
        "--execution-strategy",
        default="parallel",
        type=str,
        help="The plan executor to use. One of sequential, pipelined, parallel",
    )
    parser.add_argument(
        "--sentinel-execution-strategy",
        default="mab",
        type=str,
        help="The sentinel execution strategy to use. One of mab, all or bay",
    )
    parser.add_argument(
        "--policy",
        default="maxquality",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
    )
    parser.add_argument(
        "--val-examples",
        default=25,
        type=int,
        help="Number of validation examples to sample from",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        type=str,
        help="One of 'gpt-4o', 'gpt-4o-mini', 'llama'",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed used to initialize RNG for MAB sampling algorithm",
    )
    parser.add_argument(
        "--k",
        default=10,
        type=int,
        help="Number of columns to sample in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--j",
        default=3,
        type=int,
        help="Number of columns to sample in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--sample-budget",
        default=5,
        type=int,
        help="Total sample budget in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        type=str,
        help="The experiment name.",
    )
    parser.add_argument(
        "--priors-file",
        default=None,
        type=str,
        help="A file with a dictionary mapping physical operator ids to prior belief on their performance",
    )
    parser.add_argument(
        "--quality",
        default=None,
        type=float,
        help="Quality threshold",
    )

    # =============== NEW FEATURE ===============
    # Args da pesquisa (Axia). Defaults = valores originais do script.
    parser.add_argument("--max-workers", default=64, type=int, help="Threads concorrentes (requisicoes LLM em voo). Original: 64")
    parser.add_argument("--test-examples", default=10, type=int, help="Tamanho do split de teste. Original: 10")
    parser.add_argument("--api-base", default="http://localhost:8000/v1", type=str, help="LLMs via scheduler (igual ao CUAD)")
    parser.add_argument("--embed-api-base", default="http://localhost:8001/v1", type=str, help="Embeddings (vllm-axia.sh, porta 8001)")
    parser.add_argument("--embed-model", default="nomic-ai/nomic-embed-text-v1", type=str, help="Modelo de embedding do indice")
    # =============== END NEW FEATURE ===============

    args = parser.parse_args()

    # create directory for profiling data
    os.makedirs("opt-profiling-data", exist_ok=True)

    # parse args
    verbose = args.verbose
    progress = args.progress
    seed = args.seed
    val_examples = args.val_examples
    k = args.k
    j = args.j
    sample_budget = args.sample_budget
    execution_strategy = args.execution_strategy
    sentinel_execution_strategy = args.sentinel_execution_strategy
    exp_name = (
        f"biodex-final-{sentinel_execution_strategy}-k{k}-j{j}-budget{sample_budget}-seed{seed}"
        if args.exp_name is None
        else args.exp_name
    )
    priors = None
    if args.priors_file is not None:
        with open(args.priors_file) as f:
            priors = json.load(f)

     # diretorios do log
    os.makedirs("tests", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    strategy = sentinel_execution_strategy
    budget = sample_budget

    log_filename = (
        f"biodex_{timestamp}"
        f"_strategy-{strategy}"
        f"_budget-{budget}.log"
    )

    log_path = os.path.join("tests", log_filename)

    #policy = pz.MinCost() 
    #print(f"USING POLICY (FORCED): {policy}")
    # set the optimization policy; constraint set to 25% percentile from unconstrained plans
    # policy = pz.MaxQuality() if not args.constrained else pz.MaxQualityAtFixedCost(max_cost=2.250)
    # if args.quality is not None and args.policy == "mincostatfixedquality":
    #     policy = pz.MinCostAtFixedQuality(min_quality=args.quality)
    # elif args.quality is not None and args.policy == "minlatencyatfixedquality":
    #     policy = pz.MinTimeAtFixedQuality(min_quality=args.quality)
    policy = pz.MaxQuality() # Em vez de qualidade fixa
    print(f"USING POLICY: {policy}")

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None and os.getenv("ANTHROPIC_API_KEY") is None:
        print("WARNING: OPENAI_API_KEY, TOGETHER_API_KEY, and ANTHROPIC_API_KEY are unset")

    # create validator
    validator = BiodexValidator(
        rp_at_k=5,
        num_samples=val_examples,
        shuffle=True,
        seed=seed,
    )

    # create train dataset for validator
    train_dataset = BiodexDataset(
        split="train",
        num_samples=val_examples,
        shuffle=True,
        seed=seed,
    )
    train_dataset = {train_dataset.id: train_dataset}


    # load index [text-embedding-3-small]
    chroma_client = chromadb.PersistentClient(".chroma-biodex")
    openai_ef = OpenAIEmbeddingFunction(
        # =============== NEW FEATURE ===============
        # Embeddings do vllm-axia.sh (porta 8001); a 9000 era do ambiente local.
        api_key="EMPTY",
        model_name=args.embed_model,
        api_base=args.embed_api_base,
        # (original)
        # api_key="EMPTY",
        # model_name="nomic-ai/nomic-embed-text-v1",
        # api_base="http://localhost:9000/v1",
        # =============== END NEW FEATURE ===============
    )

    index = chroma_client.get_collection("biodex-reaction-terms", embedding_function=openai_ef)
    # =============== NEW FEATURE ===============
    # Depuracao: o retrieve engole falhas de embedding e devolve lista vazia (RP@5 cai em silencio).
    # Falhar aqui e barato; falhar depois da otimizacao custa GPU. Tambem acusa indice com
    # dimensao errada (text-embedding-3-small = 1536 vs nomic = 768).
    try:
        _n_terms = index.count()
        _probe = index.query(query_embeddings=openai_ef(["headache"]), n_results=3)
        print(f"[preflight] indice OK: {_n_terms} termos; probe('headache') -> {_probe['documents'][0]}")
    except Exception as _e:
        sys.exit(f"[erro] preflight do indice/embeddings falhou ({args.embed_api_base}): {_e}")
    if _n_terms == 0:
        sys.exit("[erro] colecao Chroma vazia")
    # =============== END NEW FEATURE ===============


    def search_func(index: chromadb.Collection, query: list[list[float]], k: int) -> list[str]:
        # execute query with embeddings
        results = index.query(query, n_results=5)

        # get list of result terms with their cosine similarity scores
        final_results = []
        for query_docs, query_distances in zip(results["documents"], results["distances"]):
            for doc, dist in zip(query_docs, query_distances):
                cosine_similarity = 1 - dist
                final_results.append({"content": doc, "similarity": cosine_similarity})

        # sort the results by similarity score
        sorted_results = sorted(final_results, key=lambda result: result["similarity"], reverse=True)

        # remove duplicates
        sorted_results_set = set()
        final_sorted_results = []
        for result in sorted_results:
            if result["content"] not in sorted_results_set:
                sorted_results_set.add(result["content"])
                final_sorted_results.append(result["content"])

        # return the top-k similar results and generation stats
        return {"reaction_labels": final_sorted_results[:k]}

    # construct plan
    # =============== NEW FEATURE ===============
    plan = BiodexDataset(split="test", num_samples=args.test_examples, shuffle=True, seed=seed)
    # (original)
    # plan = BiodexDataset(split="test", num_samples=10, shuffle=True, seed=seed) 
    # =============== END NEW FEATURE ===============
    plan = plan.sem_map(biodex_reactions_cols)
    plan = plan.retrieve(
        index=index,
        search_func=search_func,
        search_attr="reactions",
        output_attrs=biodex_reaction_labels_cols,
    )
    plan = plan.sem_map(biodex_ranked_reactions_labels_cols, depends_on=["title", "abstract", "fulltext", "reaction_labels"])

    # execute pz plan
    
    config = pz.QueryProcessorConfig(
        available_models=[
        #"hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct",
        # =============== NEW FEATURE ===============
        # Mesmos modelos do CUAD (servidos pelo scheduler na Axia).
            # MODELS_AXIA (mesmos model names do cuad-demo1.py => mesmos custos do constants.py)
            "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
            "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            # "hosted_vllm/Qwen/Qwen3-235B-A22B",
            "hosted_vllm/openai/gpt-oss-120b",
        # (original)
        #  "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
        # =============== END NEW FEATURE ===============
        # "hosted_vllm/Qwen/Qwen3-0.6B",
        # "hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
        # "hosted_vllm/Qwen/Qwen3-4B",
        #"hosted_vllm/Qwen/Qwen2.5-3B-Instruct",
        # "hosted_vllm/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        # "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct",
        # "hosted_vllm/google/gemma-3-270m",
        # "hosted_vllm/Qwen/Qwen3-4B-Instruct-2507",
        # "hosted_vllm/qwen15b_tuned",
        # "hosted_vllm/qwen3b_tuned",
        ],
        # =============== NEW FEATURE ===============
        # LLMs via scheduler (:8000), igual ao CUAD.
        api_base=args.api_base,
        # (original)
        # api_base="http://localhost:9000/v1",
        # =============== END NEW FEATURE ===============
        policy=policy,
        #optimizer_strategy="pareto",
        optimizer_strategy="greedy",
        sentinel_execution_strategy=sentinel_execution_strategy,
        execution_strategy=execution_strategy,
        use_final_op_quality=True,
        # =============== NEW FEATURE ===============
        max_workers=args.max_workers,
        # (original)
        # max_workers=64,
        # =============== END NEW FEATURE ===============
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
        # tie_percentile=0.25,
    )
    

    # =============== NEW FEATURE ===============
    # Masterfile: reserva masterfile/runs/<run_id>/ e grava run_config.json (ver masterfile/hooks.py).
    # POSICAO: depende de exp_name e config ja definidos.
    from masterfile.hooks import MasterfileRun
    mf = MasterfileRun.start(
        exp_name=exp_name,
        benchmark="biodex",
        args=args,
        pz_config=config,
        extra={
            "benchmark_variant": "maxquality-greedy",
            "script": os.path.basename(__file__),
            "max_cost": None,
            "use_final_op_quality": getattr(config, "use_final_op_quality", None),
            "test_split_size": args.test_examples,
            "train_split_size": val_examples,
            "validator_num_samples": val_examples,
            "output_dir": "opt-profiling-data",
        },
    )

    # contagem de arquivos de erro gravados silenciosamente pelo PZ/validator
    _SILENT_DIRS = ["topk-errors", "retrieve-errors", "rp@k-errors", "term-recall-eval-errors"]
    def _count_silent_errors():
        return sum(len(os.listdir(d)) for d in _SILENT_DIRS if os.path.isdir(d))
    _silent_before = _count_silent_errors()
    # =============== END NEW FEATURE ===============

    data_record_collection = plan.optimize_and_run(config=config, train_dataset=train_dataset, validator=validator)

    print(data_record_collection.to_df())
    data_record_collection.to_df().to_csv(f"opt-profiling-data/{exp_name}-output.csv", index=False)

    # create filepaths for records and stats
    records_path = f"opt-profiling-data/{exp_name}-records.json"
    stats_path = f"opt-profiling-data/{exp_name}-profiling.json"

    # save record outputs
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

    # save statistics
    execution_stats_dict = data_record_collection.execution_stats.to_json()
    with open(stats_path, "w") as f:
        json.dump(execution_stats_dict, f)

    # score output
    test_dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions", split="test").to_pandas()
    # =============== NEW FEATURE ===============
    # Mesmo n e mesma seed do plano => mesmos pmids.
    test_dataset = test_dataset.sample(n=args.test_examples, random_state=seed).to_dict(orient="records")
    # (original)
    # test_dataset = test_dataset.sample(n=250, random_state=seed).to_dict(orient="records")
    # =============== END NEW FEATURE ===============

    # construct mapping from pmid --> label (field, value) pairs
    def compute_target_record(entry):
        reactions_lst = [
            reaction.strip().lower().replace("'", "").replace("^", "")
            for reaction in entry["reactions"].split(",")
        ]
        label_dict = {"ranked_reaction_labels": reactions_lst}
        return label_dict

    label_fields_to_values = {
        entry["pmid"]: compute_target_record(entry) for entry in test_dataset
    }

    def rank_precision_at_k(preds: list, targets: list, k: int):
        if preds is None:
            return 0.0

        # lower-case each list
        preds = [pred.lower().replace("'", "").replace("^", "") for pred in preds]
        targets = set([target.lower().replace("'", "").replace("^", "") for target in targets])

        # compute rank-precision at k
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
    # =============== NEW FEATURE ===============
    # Diagnosticos de falha silenciosa. O rp@5 original e media so sobre registros que voltaram;
    # n_missing mostra quantos se perderam.
    _returned_pmids = {str(r.get("pmid")) for r in record_jsons}
    _n_expected = len(label_fields_to_values)
    _empty = sum(1 for r in record_jsons if not r.get("reaction_labels"))
    stats_dict.update({
        "n_expected": _n_expected,
        "n_returned": len(record_jsons),
        "n_missing": _n_expected - len(_returned_pmids),
        "empty_reaction_labels_frac": (_empty / len(record_jsons)) if record_jsons else None,
        "silent_error_files": _count_silent_errors() - _silent_before,
    })
    if stats_dict["n_missing"] > 0 or (stats_dict["empty_reaction_labels_frac"] or 0) > 0.2 or stats_dict["silent_error_files"] > 0:
        print("AVISO: possivel falha silenciosa (registros faltando / retrieve vazio) -- ver *-errors/")
    # =============== END NEW FEATURE ===============
    with open(f"opt-profiling-data/{exp_name}-metrics.json", "w") as f:
        json.dump(stats_dict, f)

    # =============== NEW FEATURE ===============
    # Masterfile: run.json + CSVs. Nunca levanta excecao (ver masterfile/hooks.py).
    mf.finish(execution_stats_dict, metrics=stats_dict)
    # =============== END NEW FEATURE ===============


    # adicionar o budget e qual sentinel execution strategy
    
    #print(f"bad: {bad}")
    #print("-------")
    #print(f"rp@k: {rp_at_k:.5f}")
    #print(f"Optimization time: {data_record_collection.execution_stats.optimization_time}")
    #print(f"Optimization cost: {data_record_collection.execution_stats.optimization_cost}")
    #print(f"Plan Exec. time: {data_record_collection.execution_stats.plan_execution_time}")
    #print(f"Plan Exec. cost: {data_record_collection.execution_stats.plan_execution_cost}")
    #print(f"Total Execution time: {data_record_collection.execution_stats.total_execution_time}")
    #print(f"Total Execution Cost: {data_record_collection.execution_stats.total_execution_cost}")


log("========== EXPERIMENT RESULTS ==========")
log(f"Experiment name: {exp_name}")
log(f"Sentinel execution strategy: {sentinel_execution_strategy}")
log(f"Execution strategy: {execution_strategy}")
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

# =============== NEW FEATURE ===============
# Mostra o plano final selecionado no log, no mesmo formato do cuad-demo1.py.
log("======= FINAL PLAN SELECTED =======")
log(final_plan_str)
log("===================================")
# =============== END NEW FEATURE ===============
