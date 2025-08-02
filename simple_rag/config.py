# simple_rag/config.py
"""
Configuration settings for the Simple Local RAG application.
"""

# --- Qdrant Settings ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# --- PDF Processing Settings ---
# We no longer need a default PDF path, but we'll keep the download URL as a fallback.
PDF_DOWNLOAD_URL = "https://pressbooks.oer.hawaii.edu/humannutrition2/wp-content/uploads/sites/27/2019/01/Human-Nutrition-2.pdf"
PAGE_OFFSET = 41 # Adjust if your PDF has front matter/cover pages

# --- Production-Level Chunking Settings ---
CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 30
MIN_TOKEN_LENGTH = 20

# --- Embedding Model Settings ---
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# --- LLM Settings ---
AVAILABLE_MODELS = [
    {
        "id": "google/gemma-2b-it",
        "type": "causal",
        "ram": "~5GB",
        "required_packages": ["bitsandbytes"]
    },
    {
        "id": "microsoft/phi-2",
        "type": "causal",
        "ram": "~6GB",
        "required_packages": ["bitsandbytes"]
    },
    {
        "id": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        "type": "gptq",
        "ram": "~5GB",
        "required_packages": ["optimum", "auto-gptq"]
    },
    {
        "id": "google/gemma-7b-it",
        "type": "causal",
        "ram": "~15GB",
        "required_packages": ["bitsandbytes"]
    }
]
DEFAULT_MODEL_CONFIG = AVAILABLE_MODELS[0]
MAX_CONTEXT_CHARS = 6000

# --- Retrieval Settings ---
N_RESOURCES_TO_RETURN = 5

# --- File Paths ---
LOG_FILE_PATH = "simple_local_rag.log"
TOKEN_DISTRIBUTION_SAVE_PATH = "token_distribution.png"