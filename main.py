# main.py
from simple_rag.pipeline import SimpleLocalRAG
from rich.traceback import install as install_rich_traceback
import sys

def main():
    """Main entry point for the Simple Local RAG application."""
    install_rich_traceback()
    try:
        app = SimpleLocalRAG()
        app.run_pipeline()
    except KeyboardInterrupt:
        print("\n[bold yellow]Process interrupted by user. Exiting...[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()