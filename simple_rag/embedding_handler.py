# simple_rag/embedding_handler.py
import torch
from sentence_transformers import SentenceTransformer, util

from . import utils

def create_embeddings(chunks: list, embedding_model_name: str) -> torch.Tensor:
    """Creates embeddings for a list of text chunks."""
    utils.console.print("\n[bold blue]Creating Embeddings...[/bold blue]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(model_name_or_path=embedding_model_name, device=device)
    
    text_chunks = [item["sentence_chunk"] for item in chunks]
    
    embeddings = embedding_model.encode(
        text_chunks,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    
    utils.console.print(f"[green]✓ Embeddings created with shape: {embeddings.shape}[/green]")
    return embeddings, embedding_model

def retrieve_relevant_resources(query: str, embeddings: torch.Tensor, embedding_model, n_resources: int) -> tuple:
    """Retrieves the most relevant resources for a query."""
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    
    # Ensure n_resources is not greater than the number of embeddings
    k = min(n_resources, len(embeddings))
    
    scores, indices = torch.topk(input=dot_scores, k=k)
    return scores, indices