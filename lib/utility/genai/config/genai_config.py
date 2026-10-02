import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

GENAI_DATASET_DIR = PROJECT_ROOT / "datasets" / "GEN AI"
GENAI_EVAL_DIR = GENAI_DATASET_DIR / "eval"
GENAI_ARTIFACT_DIR = PROJECT_ROOT / "saved_models" / "genai"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

HR_POLICY_PDF = GENAI_DATASET_DIR / "the_nestle_hr_policy_pdf_2012.pdf"
HR_POLICY_QA = GENAI_EVAL_DIR / "hr_policy_qa.json"
TOURISM_XLSX = GENAI_DATASET_DIR / "Capstone 2" / "Part 2" / "tourism_with_id.xlsx"


@dataclass
class GenAIConfig:
    """
    Central configuration for every GenAI utility.

    Every value can be overridden with an environment variable
    (see `from_env`) so scripts never hard-code model names.
    """

    # ======================================================
    # Ollama
    # ======================================================

    ollama_host: str = "http://localhost:11434"
    chat_model: str = "qwen3:8b"
    embedding_model: str = "nomic-embed-text"
    request_timeout: float = 600.0

    # ======================================================
    # Generation
    # ======================================================

    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 256
    context_window: int = 8192

    # qwen3 is a "thinking" model. Disabling thinking keeps
    # answers short and fast for learning scripts.
    think: bool = False

    # ======================================================
    # Chunking / Retrieval
    # ======================================================

    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 4
    fetch_k: int = 12

    # ======================================================
    # Paths
    # ======================================================

    dataset_dir: Path = field(default_factory=lambda: GENAI_DATASET_DIR)
    artifact_dir: Path = field(default_factory=lambda: GENAI_ARTIFACT_DIR)

    @property
    def vectorstore_dir(self) -> Path:
        return self.artifact_dir / "vectorstores"

    @property
    def embedding_cache_dir(self) -> Path:
        return self.artifact_dir / "embeddings_cache"

    @property
    def checkpoint_dir(self) -> Path:
        return self.artifact_dir / "checkpoints"

    @classmethod
    def from_env(cls, **overrides) -> "GenAIConfig":
        env_map = {
            "ollama_host": "OLLAMA_HOST",
            "chat_model": "GENAI_CHAT_MODEL",
            "embedding_model": "GENAI_EMBEDDING_MODEL",
        }
        values = {}
        for attr, env in env_map.items():
            if os.getenv(env):
                values[attr] = os.environ[env]
        values.update(overrides)
        return cls(**values)
