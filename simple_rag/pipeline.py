# simple_rag/pipeline.py
import os
import re
import textwrap
from rich.panel import Panel
from rich.markdown import Markdown

from . import config, utils, pdf_processor, embedding_handler, llm_handler, vector_db_handler

class SimpleLocalRAG:
    def __init__(self):
        self.collection_name = None
        self.embedding_model = None
        self.tokenizer = None
        self.llm_model = None
        self.logger = utils.setup_logging()

    def run_pipeline(self):
        """Runs the full RAG pipeline with Qdrant."""
        utils.setup_logging()
        utils.verify_dependencies()
        utils.check_cuda_support()
        
        # --- Interactive PDF Selection ---
        pdf_path = utils.select_pdf_path()
        if not pdf_path:
            self.logger.error("No PDF selected. Exiting.")
            return
            
        # --- VectorDB and PDF Processing ---
        # Create a unique, clean collection name from the PDF filename
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', pdf_name)
        
        # Check if the collection needs to be created/populated
        collection_count = vector_db_handler.get_collection_count(self.collection_name)
        
        if collection_count == 0:
            self.logger.info(f"Qdrant collection '{self.collection_name}' is empty. Processing PDF.")
            chunks = pdf_processor.process_pdf(pdf_path)
            if not chunks:
                self.logger.error("No chunks were created. Exiting.")
                return

            embeddings, self.embedding_model = embedding_handler.create_embeddings(
                chunks, config.EMBEDDING_MODEL_NAME
            )
            if embeddings is None:
                self.logger.error("Embedding creation failed. Exiting.")
                return

            # Create the collection and upsert the data
            vector_db_handler.create_or_get_collection(self.collection_name, embeddings.shape[1])
            vector_db_handler.upsert_data(self.collection_name, chunks, embeddings)
            utils.visualize_token_distribution(chunks)
        else:
            self.logger.info(f"Collection '{self.collection_name}' already populated. Skipping processing.")
            utils.console.print(f"[green]✓ Qdrant collection '{self.collection_name}' loaded with {collection_count} documents.[/green]")
            _, self.embedding_model = embedding_handler.create_embeddings([], config.EMBEDDING_MODEL_NAME)


        # --- LLM Loading ---
        selected_model_config = llm_handler.select_model()
        self.tokenizer, self.llm_model = llm_handler.load_llm(selected_model_config)
        
        # --- Start Interactive Session ---
        self.interactive_query()

    def interactive_query(self):
        """Handles the interactive user query session with Qdrant."""
        utils.console.rule(f"[bold blue]Interactive Query for '{self.collection_name}'[/bold blue]")
        while True:
            query = utils.console.input("[bold cyan]Enter your question (or 'exit'): [/bold cyan]")
            if query.lower() == 'exit':
                break
            if not query.strip():
                continue

            search_results = vector_db_handler.query_qdrant(
                self.collection_name, query, self.embedding_model, config.N_RESOURCES_TO_RETURN
            )
            
            contexts = [result.payload['text'] for result in search_results]

            utils.console.print("\n[bold blue]Top relevant passages:[/bold blue]")
            for i, result in enumerate(search_results):
                panel = Panel(
                    textwrap.fill(result.payload['text'], 100),
                    title=f"[bold]Result {i+1} (Score: {result.score:.4f}, Page: {result.payload['page_number']})[/bold]",
                    border_style="blue"
                )
                utils.console.print(panel)
            
            answer = llm_handler.generate_answer(query, contexts, self.tokenizer, self.llm_model)
            
            answer_panel = Panel(Markdown(answer), title="[bold green]Generated Answer[/bold green]", border_style="green")
            utils.console.print(answer_panel)