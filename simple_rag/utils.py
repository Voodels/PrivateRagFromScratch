# simple_rag/utils.py
import os
import sys
import re
import logging
from typing import List, Tuple
from pathlib import Path

import torch
import fitz # PyMuPDF
import spacy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text

from . import config

# Initialize console
console = Console()

def select_pdf_path() -> str:
    """
    Scans for PDF files in the current directory and subdirectories
    and prompts the user to select one.
    """
    console.rule("[bold blue]Select a PDF Document[/bold blue]")
    pdf_files = list(Path('.').rglob('*.pdf')) # rglob scans recursively
    
    if not pdf_files:
        console.print("[bold red]No PDF files found in the current directory or subdirectories.[/bold red]")
        # You could add logic here to download a default file if needed
        return None

    table = Table(title="Available PDF Files")
    table.add_column("Index", style="cyan", justify="center")
    table.add_column("File Path")
    table.add_column("Size (MB)")
    
    for i, pdf_file in enumerate(pdf_files):
        size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
        table.add_row(str(i + 1), str(pdf_file), f"{size_mb:.2f}")
    
    console.print(table)
    console.print("Select a file by number, or enter a full path to another PDF file.")
    
    while True:
        choice = console.input("[bold yellow]> [/bold yellow]")
        if choice.isdigit() and 1 <= int(choice) <= len(pdf_files):
            return str(pdf_files[int(choice) - 1])
        elif Path(choice).is_file() and choice.lower().endswith('.pdf'):
            return choice
        else:
            console.print("[bold red]Invalid selection. Please enter a valid number or path.[/bold red]")


def setup_logging():
    # ... (rest of the functions are the same as before)
    """Configures logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE_PATH),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("SimpleLocalRAG")

def verify_dependencies() -> bool:
    """Verifies that all required dependencies are available."""
    try:
        import sentence_transformers
        import transformers
        import qdrant_client
        console.print("[green]✓ All major dependencies seem to be installed.[/green]")
        return True
    except ImportError as e:
        console.print(f"[bold red]Error: Missing dependency -> {e.name}.[/bold red]")
        console.print("[yellow]Please run: pip install -r requirements.txt[/yellow]")
        return False

def check_cuda_support():
    """Checks for CUDA support and displays hardware information."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        console.print(f"[green]✓ CUDA is available - Using GPU: {device_name} ({gpu_memory:.1f} GB)[/green]")
    else:
        console.print("[yellow]⚠ CUDA not available - Using CPU (This will be significantly slower).[/yellow]")

def text_formatter(text: str) -> str:
    """Formats text from a PDF by removing extra whitespace and newlines."""
    return re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', text)).strip()

def split_list(input_list: List, slice_size: int) -> List[List]:
    """Splits a list into chunks of a specified size."""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def visualize_token_distribution(chunks: List[dict]):
    """Visualizes token distribution across chunks and saves it to a file."""
    try:
        df = pd.DataFrame(chunks)
        plt.figure(figsize=(10, 6))
        plt.hist(df["chunk_token_count"], bins=30, color='skyblue', edgecolor='black')
        plt.title("Distribution of Token Counts Across Chunks")
        plt.xlabel("Token Count per Chunk")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(config.TOKEN_DISTRIBUTION_SAVE_PATH)
        plt.close()
        console.print(f"\n[green]Token distribution visualization saved to {config.TOKEN_DISTRIBUTION_SAVE_PATH}[/green]")
    except Exception as e:
        console.print(f"[yellow]Could not create visualization: {e}[/yellow]")