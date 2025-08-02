# simple_rag/vector_db_handler.py
from qdrant_client import QdrantClient, models
import torch
from rich.progress import track

from . import utils, config

# Initialize a Qdrant client that connects to your local Docker instance
# We increase the timeout to handle potentially larger requests
client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=60)

def create_or_get_collection(collection_name: str, vector_size: int):
    """Creates a new collection in Qdrant if it doesn't exist."""
    utils.console.print(f"[bold blue]Initializing Qdrant collection: '{collection_name}'[/bold blue]")
    try:
        client.get_collection(collection_name=collection_name)
        utils.console.print(f"[green]✓ Collection '{collection_name}' already exists.[/green]")
    except Exception:
        utils.console.print(f"Collection '{collection_name}' not found. Creating new collection...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        utils.console.print(f"[green]✓ Collection '{collection_name}' created.[/green]")
    return collection_name

def upsert_data(collection_name: str, chunks: list, embeddings: torch.Tensor):
    """
    Upserts (uploads) data to the Qdrant collection in batches to prevent timeouts.
    """
    utils.console.print(f"[bold blue]Upserting data to Qdrant collection '{collection_name}'...[/bold blue]")
    
    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            models.PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={
                    "page_number": chunk['page_number'],
                    "text": chunk['sentence_chunk']
                }
            )
        )
    
    # --- BATCHING LOGIC ---
    # Upsert data in smaller batches to avoid timeouts with large datasets.
    batch_size = 256 # You can adjust this size based on your machine's performance
    for i in track(range(0, len(points), batch_size), description="Upserting batches..."):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True # Wait for the operation to complete before the next batch
        )

    utils.console.print("[green]✓ Data upserted to Qdrant successfully.[/green]")


def query_qdrant(collection_name: str, query: str, embedding_model, n_results: int) -> list:
    """Queries the Qdrant collection to find the most relevant chunks."""
    query_vector = embedding_model.encode(query, convert_to_tensor=True).tolist()
    
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=n_results,
    )
    
    return search_results

def get_collection_count(collection_name: str) -> int:
    """Gets the number of vectors in a collection."""
    try:
        return client.get_collection(collection_name=collection_name).vectors_count
    except Exception:
        return 0