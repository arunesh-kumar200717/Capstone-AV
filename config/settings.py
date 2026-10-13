import os
from pathlib import Path
from typing import Any


def load_dotenv(dotenv_path: Path, override: bool = False) -> None:
    """Load basic KEY=VALUE entries from a dotenv file."""
    if not dotenv_path.is_file():
        return

    for line in dotenv_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        key, separator, value = line.partition('=')
        if not separator or not key.strip():
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env', override=False)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data'
PAPERS_DIR = DATA_DIR / 'papers'
PROCESSED_DIR = DATA_DIR / 'processed'
EVALUATION_DIR = DATA_DIR / 'evaluation'
ARTIFACTS_DIR = ROOT_DIR / 'artifacts'
CHROMA_DIR = Path(os.getenv('CHROMA_PERSIST_DIR', './artifacts/chroma')).resolve() if os.getenv('CHROMA_PERSIST_DIR') else ROOT_DIR / 'artifacts' / 'chroma'

for directory in [PAPERS_DIR, PROCESSED_DIR, EVALUATION_DIR, ARTIFACTS_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key, str(default)).strip().lower()
    return value in {'1', 'true', 'yes', 'y', 'on'}


def get_setting(key: str, default: Any = None) -> Any:
    value = os.getenv(key)
    if value is None:
        return default
    return value


EMBEDDING_MODELS = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'BAAI/bge-small-en-v1.5',
]
DEFAULT_EMBEDDING_MODEL = get_setting('EMBEDDING_MODEL', EMBEDDING_MODELS[0])
CHUNK_SIZE = int(get_setting('CHUNK_SIZE', 800))
CHUNK_OVERLAP = int(get_setting('CHUNK_OVERLAP', 120))
TOP_K = int(get_setting('TOP_K', 3))
LLM_PROVIDER = get_setting('LLM_PROVIDER', 'openai_compatible')
LLM_MODEL = get_setting('LLM_MODEL', 'gpt-4o-mini')
OPENAI_API_KEY = get_setting('OPENAI_API_KEY')
OPENAI_BASE_URL = get_setting('OPENAI_BASE_URL')
OLLAMA_BASE_URL = get_setting('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = get_setting('OLLAMA_MODEL', '')
