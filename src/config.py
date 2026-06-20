"""
Configuration management for Telecom RAG system.
Centralized config for all components.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Module-level constants (spec-mandated) ────────────────────────────────────
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL   = "cross-encoder/mmarco-MiniLMv2-L12-H384-v1"
LLM_MODEL        = "mixtral-8x7b-32768"
TOP_K_RETRIEVE   = 20   # candidates fetched by hybrid retrieval
TOP_K_RERANK     = 5    # chunks kept after cross-encoder re-ranking
MIN_CHUNK_WORDS  = 50
BATCH_SIZE       = 32
LORA_RANK        = 16
LORA_ALPHA       = 32

DATA_DIR         = PROJECT_ROOT / "data"
RAW_DIR          = DATA_DIR / "raw"
PROCESSED_DIR    = DATA_DIR / "processed"
VECTORSTORE_PATH = DATA_DIR / "vectorstore"
MODELS_DIR       = PROJECT_ROOT / "models"
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = EMBEDDING_MODEL
    device: str = "cpu"  # or "cuda"
    batch_size: int = BATCH_SIZE
    normalize_embeddings: bool = True
    max_memory_usage: float = 0.85


@dataclass
class RetrieverConfig:
    """Retrieval configuration."""
    top_k: int = TOP_K_RETRIEVE
    top_k_rerank: int = TOP_K_RERANK
    similarity_threshold: float = 0.3
    use_hybrid: bool = True
    use_reranking: bool = True
    use_source_diversity: bool = True


@dataclass
class LLMConfig:
    """LLM configuration."""
    model_name: str = "mixtral-8x7b-32768"
    api_key: str = ""
    device: str = "cpu"
    quantize: bool = False
    max_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.95


@dataclass
class RerankerConfig:
    """Re-ranker configuration."""
    model_name: str = RERANKER_MODEL
    top_k: int = TOP_K_RERANK
    min_score: float = 0.0


@dataclass
class LoRAConfig:
    """QLoRA fine-tuning configuration."""
    rank: int = LORA_RANK
    alpha: int = LORA_ALPHA
    dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj")
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    """Data paths configuration."""
    raw_3gpp_dir: Path = Path("data/raw/3gpp_docs")
    raw_teleqna_dir: Path = Path("data/raw/teleqna")
    raw_oran_dir: Path = Path("data/raw/oran_datasets")
    raw_simu5g_dir: Path = Path("data/raw/simu5g")
    processed_dir: Path = Path("data/processed")
    vectorstore_dir: Path = Path("data/vectorstore")

    def __post_init__(self):
        """Convert to absolute paths."""
        self.raw_3gpp_dir = PROJECT_ROOT / self.raw_3gpp_dir
        self.raw_teleqna_dir = PROJECT_ROOT / self.raw_teleqna_dir
        self.raw_oran_dir = PROJECT_ROOT / self.raw_oran_dir
        self.raw_simu5g_dir = PROJECT_ROOT / self.raw_simu5g_dir
        self.processed_dir = PROJECT_ROOT / self.processed_dir
        self.vectorstore_dir = PROJECT_ROOT / self.vectorstore_dir


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    min_chunk_size: int = 200
    max_chunk_size: int = 1000
    overlap_ratio: float = 0.1
    min_chunk_words: int = 50
    parent_context_lines: int = 3


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4


@dataclass
class RAGConfig:
    """Complete RAG system configuration."""
    embedding: EmbeddingConfig
    retriever: RetrieverConfig
    llm: LLMConfig
    reranker: RerankerConfig
    lora: LoRAConfig
    data: DataConfig
    chunking: ChunkingConfig
    api: APIConfig

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embedding": asdict(self.embedding),
            "retriever": asdict(self.retriever),
            "llm": asdict(self.llm),
            "reranker": asdict(self.reranker),
            "lora": asdict(self.lora),
            "data": {k: str(v) for k, v in asdict(self.data).items()},
            "chunking": asdict(self.chunking),
            "api": asdict(self.api),
        }

    def save(self, path: Path):
        """Save config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "RAGConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            retriever=RetrieverConfig(**data.get("retriever", {})),
            llm=LLMConfig(**data.get("llm", {})),
            reranker=RerankerConfig(**data.get("reranker", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            data=DataConfig(**{k: Path(v) for k, v in data.get("data", {}).items()}),
            chunking=ChunkingConfig(**data.get("chunking", {})),
            api=APIConfig(**data.get("api", {})),
        )

    @classmethod
    def default(cls) -> "RAGConfig":
        """Get default configuration."""
        return cls(
            embedding=EmbeddingConfig(),
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
            reranker=RerankerConfig(),
            lora=LoRAConfig(),
            data=DataConfig(),
            chunking=ChunkingConfig(),
            api=APIConfig(),
        )


class ConfigManager:
    """Manages configuration loading and caching."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize config manager."""
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Load config from file or use defaults."""
        config_path = PROJECT_ROOT / "config" / "config.json"

        if config_path.exists():
            logger.info(f"Loading config from {config_path}")
            self._config = RAGConfig.load(config_path)
        else:
            logger.info("Using default configuration")
            self._config = RAGConfig.default()

            # Save defaults
            self.save()

    def get(self) -> RAGConfig:
        """Get current configuration."""
        return self._config

    def update(self, **kwargs):
        """Update config values."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info(f"Updated {key}")

    def save(self):
        """Save current config to file."""
        config_path = PROJECT_ROOT / "config" / "config.json"
        self._config.save(config_path)

    def reset_to_defaults(self):
        """Reset to default configuration."""
        self._config = RAGConfig.default()
        logger.info("Configuration reset to defaults")


# Global config manager instance
_config_manager = ConfigManager()


def get_config() -> RAGConfig:
    """Get global configuration instance."""
    return _config_manager.get()


def save_config():
    """Save configuration."""
    _config_manager.save()


def reset_config():
    """Reset configuration to defaults."""
    _config_manager.reset_to_defaults()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Get default config
    config = get_config()

    print("Current Configuration:")
    print(json.dumps(config.to_dict(), indent=2, default=str))

    # Save config
    save_config()

    # Load from file
    config_path = PROJECT_ROOT / "config" / "config.json"
    if config_path.exists():
        loaded_config = RAGConfig.load(config_path)
        print("\n✅ Config loaded successfully")
