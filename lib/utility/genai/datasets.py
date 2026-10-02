import json

from .abstractions.document import Document
from .chunking.recursive import RecursiveCharacterChunker
from .config.genai_config import HR_POLICY_PDF, HR_POLICY_QA
from .loaders.pdf_loader import PDFLoader

HR_HEADER_PATTERNS = [r"^The Nestl[eé]\s+Human Resources Policy\s*\n\s*\d+\s*$"]

# A small built-in corpus for quick embedding / similarity experiments.
SAMPLE_SENTENCES = [
    ("The employee received a salary increase after the annual review.", "pay"),
    ("Staff compensation was raised following the performance evaluation.", "pay"),
    ("Variable pay and bonuses reward high performance.", "pay"),
    ("The company offers training programmes to develop new skills.", "learning"),
    ("Workers can attend courses to improve their knowledge.", "learning"),
    ("On-the-job coaching is the main source of learning.", "learning"),
    ("Harassment and discrimination are not tolerated at work.", "conduct"),
    ("We treat every colleague with dignity and respect.", "conduct"),
    ("The cat is sleeping on the warm windowsill.", "other"),
    ("Paris is the capital city of France.", "other"),
    ("The football match ended in a draw.", "other"),
    ("Chocolate is made from roasted cocoa beans.", "other"),
]


def load_hr_policy(mode: str = "page", include_front_matter: bool = False) -> list[Document]:
    """Load the Nestlé HR policy PDF (pages 1-2 are cover / copyright)."""
    docs = PDFLoader(HR_POLICY_PDF, mode="page", remove_patterns=HR_HEADER_PATTERNS).load()
    if not include_front_matter:
        docs = [d for d in docs if d.metadata["page"] > 2]
    if mode == "single":
        return [Document("\n\n".join(d.page_content for d in docs), {"source": HR_POLICY_PDF.name, "type": "pdf"})]
    return docs


def hr_policy_text() -> str:
    return load_hr_policy(mode="single")[0].page_content


def hr_policy_chunks(chunk_size: int = 600, chunk_overlap: int = 100) -> list[Document]:
    return RecursiveCharacterChunker(chunk_size, chunk_overlap).split_documents(load_hr_policy())


def load_hr_qa(limit: int | None = None) -> list[dict]:
    """Golden question/answer set with the PDF pages that contain each answer."""
    data = json.loads(HR_POLICY_QA.read_text(encoding="utf-8"))
    return data[:limit] if limit else data
