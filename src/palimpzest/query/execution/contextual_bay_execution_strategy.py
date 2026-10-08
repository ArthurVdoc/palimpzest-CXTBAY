import logging
import numpy as np
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from palimpzest.query.execution.execution_strategy import SentinelExecutionStrategy
from palimpzest.query.execution.bay_execution_strategy import BayesianOpFrontier, BetaPosterior, GaussianPosterior
from palimpzest.core.models import SentinelPlanStats
from palimpzest.utils.progress import create_progress_manager
from palimpzest.query.operators.scan import ScanPhysicalOp, ContextScanOp
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.physical import PhysicalOperator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  OPERATOR EMBEDDING: Extrai features contextuais de operadores
# ═══════════════════════════════════════════════════════════════

class OperatorEmbedding:
    """
    Cria uma representação vetorial de cada operador físico.
    Features incluem: tipo, modelo(s), hiperparâmetros, arquitetura.
    """
    
    # Vocabulário de tipos de operadores
    OP_TYPES = [
        "LLMConvertBonded",
        "MixtureOfAgentsConvert",
        "RAGConvert",
        "CriticAndRefineConvert",
        "MarshalAndScanDataOp",
    ]
    
    # Vocabulário de modelos conhecidos
    KNOWN_MODELS = [
        "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
        "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
        "hosted_vllm/Qwen/Qwen3-235B-A22B",  
        "hosted_vllm/openai/gpt-oss-120b",
    ]
    
    @staticmethod
    def extract(op: PhysicalOperator) -> np.ndarray:
        """
        Extrai embedding de um operador físico.
        
        Retorna vetor de features:
        - [0:n_types] one-hot do tipo de operador
        - [n_types:n_types+n_models] multi-hot dos modelos usados
        - [n_types+n_models:...] features numéricas normalizadas
        """
        features = []
        
        # 1. Tipo de operador (one-hot)
        op_class = op.__class__.__name__
        op_type_vec = np.zeros(len(OperatorEmbedding.OP_TYPES))
        if op_class in OperatorEmbedding.OP_TYPES:
            op_type_vec[OperatorEmbedding.OP_TYPES.index(op_class)] = 1.0
        features.append(op_type_vec)
        
        # 2. Modelos usados (multi-hot)
        model_vec = np.zeros(len(OperatorEmbedding.KNOWN_MODELS))
        
        # Extrai modelos do operador
        models_used = []
        if hasattr(op, 'model') and op.model:
            models_used.append(str(op.model))
        if hasattr(op, 'proposer_models'):
            models_used.extend([str(m) for m in op.proposer_models])
        if hasattr(op, 'aggregator_model'):
            models_used.append(str(op.aggregator_model))
        if hasattr(op, 'critic_model'):
            models_used.append(str(op.critic_model))
        if hasattr(op, 'refine_model'):
            models_used.append(str(op.refine_model))
        
        # Marca modelos presentes
        for model_str in models_used:
            for i, known_model in enumerate(OperatorEmbedding.KNOWN_MODELS):
                if known_model in model_str:
                    model_vec[i] = 1.0
        
        features.append(model_vec)
        
        # 3. Features numéricas normalizadas
        numeric_features = []
        
        # Temperatura (se houver)
        if hasattr(op, 'temperatures') and op.temperatures:
            numeric_features.append(np.mean(op.temperatures))
        else:
            numeric_features.append(0.0)
        
        # Número de proposers (MoA)
        if hasattr(op, 'proposer_models'):
            numeric_features.append(len(op.proposer_models) / 3.0)  # Normaliza por max=3
        else:
            numeric_features.append(0.0)
        
        # RAG parameters
        if hasattr(op, 'num_chunks_per_field'):
            numeric_features.append(op.num_chunks_per_field / 4.0)  # Normaliza por max=4
        else:
            numeric_features.append(0.0)
        
        if hasattr(op, 'chunk_size'):
            numeric_features.append(op.chunk_size / 4000.0)  # Normaliza por max=4000
        else:
            numeric_features.append(0.0)
        
        features.append(np.array(numeric_features))
        
        # Concatena tudo
        embedding = np.concatenate(features)
        
        return embedding
    
    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Computa similaridade entre dois embeddings usando kernel RBF.
        Retorna valor em [0, 1] onde 1 = idêntico.
        """
        # RBF kernel: exp(-gamma * ||x - y||^2)
        gamma = 0.1  # Controla "largura" do kernel
        dist_sq = np.sum((emb1 - emb2) ** 2)
        return np.exp(-gamma * dist_sq)


# ═══════════════════════════════════════════════════════════════
#  CONTEXTUAL PRIOR STORE: Memória persistente de operadores
# ═══════════════════════════════════════════════════════════════

class ContextualPriorStore:
    """
    Gerencia memória persistente de posteriors de operadores entre execuções.
    Armazena embeddings e posteriors em formato JSON.
    """
    
    def __init__(self, cache_dir: str = "operator_memory"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "operator_priors.json"
        
        self.memory = self._load_memory()
    
    def _load_memory(self) -> dict:
        """Carrega memória de execuções anteriores."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    memory = json.load(f)
                logger.info(f"✓ Loaded operator memory: {len(memory.get('operators', {}))} operators")
                return memory
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}")
        
        # Inicializa memória vazia
        return {
            "version": "1.0",
            "last_updated": None,
            "operators": {}
        }
    
    def save_memory(self):
        """Persiste memória em disco."""
        self.memory["last_updated"] = datetime.now().isoformat()
        
        with open(self.cache_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
        
        logger.info(f"✓ Saved operator memory: {len(self.memory['operators'])} operators")
    
    def get_prior(self, op_id: str) -> dict | None:
        """Retorna prior de um operador se existir."""
        return self.memory["operators"].get(op_id)
    
    def update_posterior(self, op_id: str, embedding: np.ndarray, posteriors: dict):
        """
        Atualiza posterior de um operador na memória.
        
        Args:
            op_id: ID do operador
            embedding: Vetor de features
            posteriors: Dict com posteriors {quality, cost, time, selectivity}
        """
        if op_id not in self.memory["operators"]:
            self.memory["operators"][op_id] = {
                "embedding": embedding.tolist(),
                "executions": 0,
                "first_seen": datetime.now().isoformat()
            }
        
        entry = self.memory["operators"][op_id]
        entry["executions"] += 1
        entry["last_seen"] = datetime.now().isoformat()
        
        # Atualiza posteriors
        entry["quality"] = {
            "alpha": posteriors["quality"].a,
            "beta": posteriors["quality"].b
        }
        entry["cost"] = {
            "mu": posteriors["cost"].mu,
            "sigma": posteriors["cost"].sigma,
            "n": posteriors["cost"].n
        }
        entry["time"] = {
            "mu": posteriors["time"].mu,
            "sigma": posteriors["time"].sigma,
            "n": posteriors["time"].n
        }
        entry["selectivity"] = {
            "alpha": posteriors["selectivity"].a,
            "beta": posteriors["selectivity"].b
        }
    
    def get_all_embeddings(self) -> tuple[list, list]:
        """
        Retorna todos embeddings e IDs armazenados.
        Usado para construir matriz de similaridade.
        """
        embeddings = []
        op_ids = []
        
        for op_id, data in self.memory["operators"].items():
            if "embedding" in data:
                embeddings.append(np.array(data["embedding"]))
                op_ids.append(op_id)
        
        return embeddings, op_ids


# ═══════════════════════════════════════════════════════════════
#  GAUSSIAN PROCESS CONTEXTUAL: Propaga conhecimento via GP
# ═══════════════════════════════════════════════════════════════

class GaussianProcessContextualPrior:
    """
    Usa Gaussian Process para interpolar priors contextuais.
    Dado um novo operador, encontra operadores similares e pondera seus posteriors.
    """
    
    def __init__(self, prior_store: ContextualPriorStore, metric: str = "quality"):
        """
        Args:
            prior_store: Store com histórico de operadores
            metric: Métrica a interpolar (quality, cost, time, selectivity)
        """
        self.prior_store = prior_store
        self.metric = metric
        
        # Carrega dados de treino (operadores conhecidos)
        self.train_embeddings, self.train_ids = prior_store.get_all_embeddings()
        
        # Extrai valores observados para a métrica
        self.train_values = []
        for op_id in self.train_ids:
            prior = prior_store.get_prior(op_id)
            if metric in ["quality", "selectivity"]:
                # Para Beta: usa mean = alpha / (alpha + beta)
                alpha = prior[metric]["alpha"]
                beta = prior[metric]["beta"]
                mean_val = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            else:  # cost, time
                # Para Gaussian: usa mu
                mean_val = prior[metric]["mu"]
            
            self.train_values.append(mean_val)
        
        self.train_values = np.array(self.train_values).reshape(-1, 1)
        
        # Inicializa GP (se houver dados de treino)
        self.gp = None
        if len(self.train_embeddings) > 0:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=3
            )
            
            # Fit GP
            X_train = np.array(self.train_embeddings)
            self.gp.fit(X_train, self.train_values.ravel())
    
    def predict_prior(self, op_embedding: np.ndarray, k_neighbors: int = 5) -> tuple[float, float]:
        """
        Prediz prior contextual para um novo operador.
        
        Returns:
            (prior_mean, prior_uncertainty): Mean e incerteza (std) do prior
        """
        if self.gp is None or len(self.train_embeddings) == 0:
            # Sem dados históricos: retorna prior não-informativo
            return 0.5, 1.0
        
        # Computa similaridades
        similarities = np.array([
            OperatorEmbedding.compute_similarity(op_embedding, train_emb)
            for train_emb in self.train_embeddings
        ])
        
        # Seleciona k vizinhos mais próximos
        top_k_idx = np.argsort(similarities)[-k_neighbors:]
        top_k_sims = similarities[top_k_idx]
        
        # Se similaridade máxima é baixa, aumenta incerteza
        max_sim = np.max(top_k_sims)
        uncertainty_factor = 1.0 + (1.0 - max_sim)  # [1.0, 2.0]
        
        # Predição via GP
        op_emb_2d = op_embedding.reshape(1, -1)
        pred_mean, pred_std = self.gp.predict(op_emb_2d, return_std=True)
        
        # Ajusta incerteza baseado em similaridade
        pred_std_adjusted = pred_std[0] * uncertainty_factor
        
        return pred_mean[0], pred_std_adjusted


# ═══════════════════════════════════════════════════════════════
#  ADAPTIVE THRESHOLD: Threshold dinâmico para fronteira Pareto
# ═══════════════════════════════════════════════════════════════

def adaptive_pareto_threshold(
    operators: list,
    posteriors: dict,
    base_thresh: float = 0.03,
    max_thresh: float = 0.10
) -> float:
    """
    Computa threshold adaptativo baseado em incerteza dos posteriors.
    
    Mais incerteza → threshold maior (mais conservador, mantém mais operadores)
    Menos incerteza → threshold menor (mais agressivo, remove operadores piores)
    
    Args:
        operators: Lista de operadores
        posteriors: Dict com posteriors de cada operador
        base_thresh: Threshold mínimo (quando confiança é alta)
        max_thresh: Threshold máximo (quando confiança é baixa)
    
    Returns:
        Threshold ajustado em [base_thresh, max_thresh]
    """
    if len(operators) == 0:
        return base_thresh
    
    # Computa incerteza média dos operadores
    uncertainties = []
    
    for op in operators:
        fid = op.get_full_op_id()
        post = posteriors[fid]
        
        # Incerteza na qualidade (Beta)
        alpha_q = post["quality"].a
        beta_q = post["quality"].b
        # Variância Beta = (alpha*beta) / ((alpha+beta)^2 * (alpha+beta+1))
        total = alpha_q + beta_q
        if total > 0:
            var_q = (alpha_q * beta_q) / (total**2 * (total + 1))
        else:
            var_q = 0.25  # Max variância para Beta uniforme
        
        # Incerteza no custo (Gaussian)
        var_c = post["cost"].sigma ** 2 / max(post["cost"].n, 1)
        
        # Média das incertezas normalizadas
        uncertainty = (var_q + var_c) / 2.0
        uncertainties.append(uncertainty)
    
    # Média das incertezas
    avg_uncertainty = np.mean(uncertainties)
    
    # Mapeia incerteza para threshold
    # Normaliza uncertainty para [0, 1] (assumindo max=0.25)
    norm_uncertainty = np.clip(avg_uncertainty / 0.25, 0, 1)
    
    # Interpola linearmente
    threshold = base_thresh + (max_thresh - base_thresh) * norm_uncertainty
    
    logger.debug(f"Adaptive threshold: {threshold:.4f} (uncertainty: {avg_uncertainty:.4f})")
    
    return threshold


# ═══════════════════════════════════════════════════════════════
#  CONTEXTUAL BAYESIAN OP FRONTIER: Frontier com contexto
# ═══════════════════════════════════════════════════════════════

class ContextualBayesianOpFrontier(BayesianOpFrontier):
    """
    Extends BayesianOpFrontier com conhecimento contextual:
    1. Inicializa posteriors com priors contextuais da memória
    2. Usa GP para interpolar priors de operadores desconhecidos
    3. Threshold adaptativo baseado em incerteza
    4. Monte Carlo denso (200 samples)
    """
    
    def __init__(self, *args, prior_store: ContextualPriorStore = None, training: bool = False, **kwargs):
        # Inicializa frontier base
        super().__init__(*args, **kwargs)
        self.prior_store = prior_store
        self.training = training  # ← controla se atualiza memória ao fim da run
        
        # Computa embeddings de todos operadores
        all_ops = self.frontier_ops + self.reservoir_ops
        self.op_embeddings = {}
        
        for op in all_ops:
            fid = op.get_full_op_id()
            self.op_embeddings[fid] = OperatorEmbedding.extract(op)
        
        # Inicializa posteriors com priors contextuais
        if prior_store:
            self._initialize_contextual_priors()
    
    def _initialize_contextual_priors(self):
        """
        Inicializa posteriors com priors contextuais da memória.
        Usa GP para interpolar priors de operadores desconhecidos.
        """
        logger.info("Initializing contextual priors...")
        
        # Constrói GPs para cada métrica
        gp_quality = GaussianProcessContextualPrior(self.prior_store, metric="quality")
        gp_cost = GaussianProcessContextualPrior(self.prior_store, metric="cost")
        gp_time = GaussianProcessContextualPrior(self.prior_store, metric="time")
        gp_selectivity = GaussianProcessContextualPrior(self.prior_store, metric="selectivity")
        
        # Para cada operador, prediz prior contextual
        for fid, embedding in self.op_embeddings.items():
            # Tenta carregar prior exato da memória
            stored_prior = self.prior_store.get_prior(fid)
            
            if stored_prior and stored_prior.get("executions", 0) > 0:
                # Operador já foi visto: usa posterior armazenado
                self.posteriors[fid]["quality"] = BetaPosterior(
                    a=stored_prior["quality"]["alpha"],
                    b=stored_prior["quality"]["beta"]
                )
                self.posteriors[fid]["selectivity"] = BetaPosterior(
                    a=stored_prior["selectivity"]["alpha"],
                    b=stored_prior["selectivity"]["beta"]
                )
                self.posteriors[fid]["cost"] = GaussianPosterior(
                    mu=stored_prior["cost"]["mu"],
                    sigma=stored_prior["cost"]["sigma"]
                )
                self.posteriors[fid]["cost"].n = stored_prior["cost"]["n"]
                
                self.posteriors[fid]["time"] = GaussianPosterior(
                    mu=stored_prior["time"]["mu"],
                    sigma=stored_prior["time"]["sigma"]
                )
                self.posteriors[fid]["time"].n = stored_prior["time"]["n"]
                
                logger.debug(f"  {fid}: Loaded from memory ({stored_prior['executions']} executions)")
            
            else:
                # Operador novo: usa GP para interpolar prior contextual
                q_mean, q_std = gp_quality.predict_prior(embedding)
                c_mean, c_std = gp_cost.predict_prior(embedding)
                t_mean, t_std = gp_time.predict_prior(embedding)
                s_mean, s_std = gp_selectivity.predict_prior(embedding)
                
                # Converte predições em posteriors
                # Para Beta: usa method of moments
                # E[Beta(a,b)] = a/(a+b) = q_mean
                # Var[Beta(a,b)] = ab/((a+b)^2(a+b+1)) = q_std^2
                # Resolve para a, b (aproximado)
                
                
                # def beta_params_from_moments(mean, var):
                #     if var >= mean * (1 - mean):
                #         # Variância muito alta: usa prior uniforme
                #         return 1.0, 1.0
                #     a = mean * (mean * (1 - mean) / var - 1)
                #     b = (1 - mean) * (mean * (1 - mean) / var - 1)
                #     return max(a, 1.0), max(b, 1.0)

                #=========REVISED UP=========
                def beta_params_from_moments(mean, var):
                    """Converte mean/var em parâmetros Beta(a, b)."""
                    # Limita variância mínima para evitar explosão
                    MIN_VAR = 0.001  # ← ADICIONAR ISTO
                    var = max(var, MIN_VAR)
                    
                    if var >= mean * (1 - mean):
                        return 1.0, 1.0
                    
                    a = mean * (mean * (1 - mean) / var - 1)
                    b = (1 - mean) * (mean * (1 - mean) / var - 1)
                    
                    # CRÍTICO: Limitar valores máximos
                    MAX_PRIOR = 100.0  # ← ADICIONAR ISTO
                    a = max(1.0, min(a, MAX_PRIOR))
                    b = max(1.0, min(b, MAX_PRIOR))
                    
                    return a, b
                #========REVISED DOWN========

                q_a, q_b = beta_params_from_moments(q_mean, q_std**2)
                s_a, s_b = beta_params_from_moments(s_mean, s_std**2)
                
                self.posteriors[fid]["quality"] = BetaPosterior(a=q_a, b=q_b)
                self.posteriors[fid]["selectivity"] = BetaPosterior(a=s_a, b=s_b)
                self.posteriors[fid]["cost"] = GaussianPosterior(mu=c_mean, sigma=c_std)
                self.posteriors[fid]["time"] = GaussianPosterior(mu=t_mean, sigma=t_std)
                
                logger.debug(f"  {fid}: Predicted via GP (q={q_mean:.3f}±{q_std:.3f})")
        
        logger.info(f"✓ Initialized {len(self.posteriors)} operators with contextual priors")
    
    def update_frontier(self, unique_logical_op_id, plan_stats):
        """
        Override do update_frontier para usar:
        1. Monte Carlo denso (200 samples)
        2. Threshold adaptativo
        """
        # Roda update base (atualiza posteriors com novas observações)
        full_op_id_to_op_stats = plan_stats.operator_stats.get(unique_logical_op_id, {})
        
        # Atualiza posteriors (copiado do bay_execution_strategy)
        means = defaultdict(lambda: defaultdict(float))
        counts = defaultdict(int)
        
        for full_op_id, op_stats in full_op_id_to_op_stats.items():
            for rec in op_stats.record_op_stats_lst:
                counts[full_op_id] += 1
                means[full_op_id]["quality"] += rec.quality or 0
                means[full_op_id]["selectivity"] += rec.passed_operator
                means[full_op_id]["cost"] += rec.cost_per_record
                means[full_op_id]["time"] += rec.time_per_record
        
        # Atualiza posteriors
        for full_op_id, cnt in counts.items():
            if cnt == 0:
                continue
            
            post = self.posteriors[full_op_id]
            m = means[full_op_id]
            
            post["quality"].update(m["quality"] / max(cnt, 1))
            post["selectivity"].update(m["selectivity"] / max(cnt, 1))
            post["cost"].update(m["cost"] / max(cnt, 1))
            post["time"].update(m["time"] / max(cnt, 1))
        
        # ═══════════════════════════════════════════════════════
        #  MONTE CARLO DENSO (200 samples)
        # ═══════════════════════════════════════════════════════
        
        MC_SAMPLES = 200  # 5x mais que original
        op_list = self.frontier_ops + self.reservoir_ops
        frontier_prob = {op.get_full_op_id(): 0 for op in op_list}
        
        # Helper: sample operator metrics via Thompson Sampling
        def sample_operator_metrics(op):
            fid = op.get_full_op_id()
            p = self.posteriors[fid]
            return {
                "quality": p["quality"].sample(),
                "selectivity": p["selectivity"].sample(),
                "cost": p["cost"].sample(),
                "time": p["time"].sample()
            }
        
        # Helper: dominance check
        def dominates(opA, opB):
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
        
        # Executa MC samples
        for _ in range(MC_SAMPLES):
            samples = {op.get_full_op_id(): sample_operator_metrics(op) for op in op_list}
            
            # Computa Pareto set neste sample
            pareto = set()
            for opA in op_list:
                A = samples[opA.get_full_op_id()]
                dominated = False
                for opB in op_list:
                    if opA is opB:
                        continue
                    B = samples[opB.get_full_op_id()]
                    if dominates(B, A):
                        dominated = True
                        break
                if not dominated:
                    pareto.add(opA.get_full_op_id())
            
            # Incrementa contadores
            for fid in pareto:
                frontier_prob[fid] += 1
        
        # Normaliza probabilidades
        for fid in frontier_prob:
            frontier_prob[fid] /= MC_SAMPLES
        
        # ═══════════════════════════════════════════════════════
        #  THRESHOLD ADAPTATIVO
        # ═══════════════════════════════════════════════════════
        
        THRESH = adaptive_pareto_threshold(
            operators=op_list,
            posteriors=self.posteriors,
            base_thresh=0.03,
            max_thresh=0.10
        )
        
        # Mantém operadores acima do threshold
        new_frontier = []
        for op in op_list:
            fid = op.get_full_op_id()
            if frontier_prob[fid] >= THRESH:
                new_frontier.append(op)
        
        # Ordena por probabilidade Pareto
        new_frontier = sorted(
            new_frontier,
            key=lambda op: frontier_prob[op.get_full_op_id()],
            reverse=True
        )
        
        # Mantém top-k
        self.frontier_ops = new_frontier[: self.k]
        remaining = [op for op in op_list if op not in self.frontier_ops]
        self.reservoir_ops = remaining
        self.off_frontier_ops = []
        
        logger.info(
            f"[Contextual Bayesian] Frontier: {len(self.frontier_ops)} ops "
            f"(threshold={THRESH:.3f}, MC_samples={MC_SAMPLES})"
        )
        
        # # Salva posteriors atualizados na memória
        # if self.prior_store:
        #     for op in self.frontier_ops + self.reservoir_ops:
        #         fid = op.get_full_op_id()
        #         self.prior_store.update_posterior(
        #             op_id=fid,
        #             embedding=self.op_embeddings[fid],
        #             posteriors=self.posteriors[fid]
        #         )
        
        # Salva posteriors atualizados na memória (apenas se em modo treino)
        if self.prior_store and self.training:
            for op in self.frontier_ops + self.reservoir_ops:
                fid = op.get_full_op_id()
                self.prior_store.update_posterior(
                    op_id=fid,
                    embedding=self.op_embeddings[fid],
                    posteriors=self.posteriors[fid]
                )
            logger.debug("[CXTBAY] Memory updated (training mode ON)")
        else:
            logger.debug("[CXTBAY] Memory NOT updated (training mode OFF)")


# ═══════════════════════════════════════════════════════════════
#  CONTEXTUAL BAYESIAN EXECUTION STRATEGY
# ═══════════════════════════════════════════════════════════════

class ContextualBayesianExecutionStrategy(SentinelExecutionStrategy):
    """
    Execution strategy com otimização bayesiana contextual.
    
    Diferenças vs BayesianExecutionStrategy:
    1. Usa ContextualBayesianOpFrontier (com GP e memória)
    2. Monte Carlo denso (200 samples)
    3. Threshold adaptativo
    4. Persiste aprendizado entre execuções
    """
    
    def __init__(self, cache_dir: str = "operator_memory", training: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training = training  # ← armazena para uso em execute_sentinel_plan
        # Inicializa store de memória
        self.prior_store = ContextualPriorStore(cache_dir=cache_dir)
        
        logger.info(
            f"[Contextual Bayesian Strategy] Initialized with "
            f"{len(self.prior_store.memory.get('operators', {}))} cached operators"
            f"| training={'ON' if training else 'OFF'}"
        )
    
    def execute_sentinel_plan(self, plan, train_dataset, validator):
        logger.info(f"Executing (Contextual Bayesian) plan {plan.plan_id}")
        
        # Initialize stats
        plan_stats = SentinelPlanStats.from_plan(plan)
        plan_stats.start()
        
        # Shuffle dataset indices
        dataset_id_to_shuffled_source_indices = {}
        for dataset_id, dataset in train_dataset.items():
            idxs = [f"{dataset_id}-{i}" for i in range(len(dataset))]
            self.rng.shuffle(idxs)
            dataset_id_to_shuffled_source_indices[dataset_id] = idxs
        
        # ═══════════════════════════════════════════════════════
        #  Cria CONTEXTUAL Bayesian frontiers
        # ═══════════════════════════════════════════════════════
        
        op_frontiers = {}
        for topo_idx, (logical_op_id, op_set) in enumerate(plan):
            unique_id = f"{topo_idx}-{logical_op_id}"
            source_unique_ids = plan.get_source_unique_logical_op_ids(unique_id)
            root_ids = plan.get_root_dataset_ids(unique_id)
            
            sample_op = op_set[0]
            
            if isinstance(sample_op, (ScanPhysicalOp, ContextScanOp)):
                assert len(root_ids) == 1
                source_idxs = dataset_id_to_shuffled_source_indices[root_ids[0]]
            
            elif isinstance(sample_op, JoinOp):
                assert len(source_unique_ids) == 2
                left = op_frontiers[source_unique_ids[0]].source_indices
                right = op_frontiers[source_unique_ids[1]].source_indices
                source_idxs = [(l, r) for l in left for r in right]
            
            else:
                assert len(source_unique_ids) == 1
                source_idxs = op_frontiers[source_unique_ids[0]].source_indices
            
            # Cria ContextualBayesianOpFrontier (com prior_store)
            op_frontiers[unique_id] = ContextualBayesianOpFrontier(
                op_set,
                source_unique_ids,
                root_ids,
                source_idxs,
                self.k,
                self.j,
                self.seed,
                self.policy,
                self.priors,
                prior_store=self.prior_store,  # ← KEY DIFFERENCE
                training=self.training  # ← propaga o flag
            )
        
        # Progress manager
        self.progress_manager = create_progress_manager(
            plan, sample_budget=self.sample_budget, progress=self.progress
        )
        self.progress_manager.start()
        
        try:
            # ═══════════════════════════════════════════════════════
            #  Sampling loop (igual ao bay_execution_strategy)
            # ═══════════════════════════════════════════════════════
            
            samples_drawn = 0
            consecutive_no_progress_iterations = 0
            max_no_progress_iterations = 5
            
            while samples_drawn < self.sample_budget:
                # Coleta índices necessários
                source_indices_to_sample = set()
                for fr in op_frontiers.values():
                    source_indices_to_sample |= fr.get_source_indices_for_next_iteration()
                
                # Executa operadores em ordem topológica
                samples_before_iteration = samples_drawn
                for topo_idx, (logical_op_id, _) in enumerate(plan):
                    unique_id = f"{topo_idx}-{logical_op_id}"
                    fr = op_frontiers[unique_id]
                    
                    # Seleciona operador com maior média posterior de qualidade
                    max_quality_op = max(
                        fr.get_frontier_ops(),
                        key=lambda op: fr.posteriors[op.get_full_op_id()]["quality"].mean
                    )
                    
                    # Pega inputs
                    inputs = fr.get_frontier_op_inputs(
                        source_indices_to_sample,
                        max_quality_op
                    )
                    inputs = [x for x in inputs if x[-1] is not None]
                    if not inputs:
                        break
                    
                    # Executa operadores
                    try:
                        out, num_ops = self._execute_op_set(unique_id, inputs)
                        samples_drawn += num_ops
                    except Exception as e:
                        logger.error(f"Error executing op set for {unique_id}: {e}")
                        break
                    
                    # Valida outputs
                    all_record_sets = {
                        idx: [(rs, op) for rs, op, _ in tuples]
                        for idx, tuples in out.items()
                    }
                    all_record_sets, val_stats = self._score_quality(
                        validator, all_record_sets
                    )
                    
                    # Atualiza plan stats
                    new_stats = []
                    for tuples in out.values():
                        for rs, _, is_new in tuples:
                            if is_new:
                                new_stats.extend(rs.record_op_stats)
                    
                    plan_stats.add_record_op_stats(unique_id, new_stats)
                    plan_stats.add_validation_gen_stats(unique_id, val_stats)
                    
                    # Alimenta próximo operador
                    next_id = plan.get_next_unique_logical_op_id(unique_id)
                    if next_id:
                        best_sets = {
                            idx: [rs for rs, _ in record_list]
                            for idx, record_list in all_record_sets.items()
                        }
                        op_frontiers[next_id].update_inputs(unique_id, best_sets)
                    
                    # ═══════════════════════════════════════════════
                    #  Atualiza fronteira (com MC denso + threshold adaptativo)
                    # ═══════════════════════════════════════════════
                    fr.update_frontier(unique_id, plan_stats)
                
                # Checa progresso
                if samples_drawn == samples_before_iteration:
                    consecutive_no_progress_iterations += 1
                    logger.warning(
                        f"No progress ({consecutive_no_progress_iterations}/{max_no_progress_iterations})"
                    )
                    if consecutive_no_progress_iterations >= max_no_progress_iterations:
                        logger.error("Stopping: no progress for 5 iterations")
                        break
                else:
                    consecutive_no_progress_iterations = 0
        
        finally:
            self.progress_manager.finish()
            
            # ═══════════════════════════════════════════════════════
            #  SALVA MEMÓRIA
            # ═══════════════════════════════════════════════════════
            # self.prior_store.save_memory()
            if self.training:
                self.prior_store.save_memory()
                logger.info("[CXTBAY] Run completed — memory persisted (training mode ON)")
            else:
                logger.info("[CXTBAY] Run completed — memory NOT updated (training mode OFF)")            
        
        plan_stats.finish()
        return plan_stats