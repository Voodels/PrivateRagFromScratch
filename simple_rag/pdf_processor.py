# simple_rag/pdf_processor.py
import fitz  # PyMuPDF
import spacy
from rich.progress import Progress,track

from . import utils
from . import config

def _clean_text(text: str) -> str:
    """
    Performs advanced cleaning of text extracted from a PDF.
    - Removes hyphenation at the end of lines.
    - Fixes common ligatures.
    - Normalizes whitespace.
    """
    # Fix hyphenation across lines (e.g., "appli- cation" -> "application")
    text = text.replace("-\n", "").replace("- ", "")
    # Fix common ligatures (optional, but good for older PDFs)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    # Use the basic formatter for newline and whitespace normalization
    return utils.text_formatter(text)


def process_pdf(pdf_path: str) -> list:
    """
    Reads a PDF and processes it into high-quality, overlapping chunks.

    This production-level function implements a sliding window chunking
    strategy to ensure that semantic context is preserved across chunks,
    which is critical for retrieval accuracy.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        list: A list of dictionaries, where each dictionary is a processed chunk.
    """
    utils.console.print(f"[bold blue]Processing PDF: {pdf_path}[/bold blue]")

    # 1. Load spaCy model for sentence splitting
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        utils.console.print("[yellow]Downloading spaCy model...[/yellow]")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # 2. Extract text and sentence metadata from PDF
    doc = fitz.open(pdf_path)
    all_sentences = []
    with Progress() as progress:
        task = progress.add_task(f"Extracting sentences from {doc.page_count} pages...", total=doc.page_count)
        for page_number, page in enumerate(doc):
            text = page.get_text()
            cleaned_text = _clean_text(text)
            
            # Use spaCy to split the text into sentences
            spacy_doc = nlp(cleaned_text)
            for sent in spacy_doc.sents:
                all_sentences.append({
                    "text": str(sent),
                    "page_number": page_number - config.PAGE_OFFSET
                })
            progress.update(task, advance=1)

    # 3. Create overlapping chunks from the sentences
    utils.console.print("[bold blue]Creating overlapping chunks...[/bold blue]")
    
    # Calculate step size for the sliding window
    step_size = config.CHUNK_SIZE_TOKENS - config.CHUNK_OVERLAP_TOKENS
    
    all_chunks = []
    current_chunk_tokens = []
    
    for sentence in track(all_sentences, description="Chunking document..."):
        # Approximate tokens in the sentence (1 token ~= 4 chars)
        sentence_tokens = len(sentence["text"]) / 4
        
        # If adding the next sentence doesn't exceed the chunk size, add it
        if sum(len(t["text"])/4 for t in current_chunk_tokens) + sentence_tokens <= config.CHUNK_SIZE_TOKENS:
            current_chunk_tokens.append(sentence)
        else:
            # --- Create a chunk ---
            chunk_text = " ".join([s["text"] for s in current_chunk_tokens])
            
            # Get the page numbers covered by this chunk
            page_numbers = sorted(list(set([s["page_number"] for s in current_chunk_tokens])))
            
            # Create a dictionary for the chunk
            chunk_dict = {
                "page_number": page_numbers[0] if len(page_numbers) == 1 else f"{page_numbers[0]}-{page_numbers[-1]}",
                "sentence_chunk": chunk_text,
                "chunk_char_count": len(chunk_text),
                "chunk_word_count": len(chunk_text.split()),
                "chunk_token_count": len(chunk_text) / 4
            }
            all_chunks.append(chunk_dict)

            # --- Start the next chunk (sliding window) ---
            # Remove tokens from the front until we are under the step size
            while sum(len(t["text"])/4 for t in current_chunk_tokens) > step_size:
                current_chunk_tokens.pop(0)
            
            # Add the current sentence to the new chunk
            current_chunk_tokens.append(sentence)

    # Add the last remaining chunk
    if current_chunk_tokens:
        chunk_text = " ".join([s["text"] for s in current_chunk_tokens])
        page_numbers = sorted(list(set([s["page_number"] for s in current_chunk_tokens])))
        chunk_dict = {
            "page_number": page_numbers[0] if len(page_numbers) == 1 else f"{page_numbers[0]}-{page_numbers[-1]}",
            "sentence_chunk": chunk_text,
            "chunk_char_count": len(chunk_text),
            "chunk_word_count": len(chunk_text.split()),
            "chunk_token_count": len(chunk_text) / 4
        }
        all_chunks.append(chunk_dict)

    # 4. Filter out any chunks that are too small
    final_chunks = [chunk for chunk in all_chunks if chunk["chunk_token_count"] > config.MIN_TOKEN_LENGTH]
    
    utils.console.print(f"[green]✓ PDF processed into {len(final_chunks)} overlapping chunks.[/green]")
    return final_chunks