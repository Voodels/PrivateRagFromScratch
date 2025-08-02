# simple_rag/llm_handler.py
import torch
import importlib
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

from . import utils, config

def _check_dependencies(model_config: dict) -> bool:
    """Checks if model-specific dependencies are installed."""
    packages = model_config.get("required_packages", [])
    all_found = True
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            utils.console.print(f"[bold red]Missing required package: '{package}'.[/bold red]")
            utils.console.print(f"[yellow]Please install it with: pip install {package}[/yellow]")
            all_found = False
    return all_found

def _load_causal_lm(model_id: str) -> tuple:
    """Loads a standard Causal Language Model with optional quantization."""
    use_quantization = True
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / 1024**3 > 20:
        use_quantization = False

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if use_quantization else None
    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )
    if not use_quantization:
        model.to("cuda")
    return tokenizer, model

def _load_gptq_lm(model_id: str) -> tuple:
    """Loads a GPTQ quantized model."""
    utils.console.print("[yellow]Loading GPTQ model. This requires 'optimum' and 'auto-gptq'.[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto", # GPTQ models handle device mapping
        trust_remote_code=False,
        revision="main"
    )
    return tokenizer, model

def select_model() -> dict:
    """Displays a menu for the user to select a model."""
    utils.console.print("\n[bold blue]Select a Language Model:[/bold blue]")
    table = utils.Table(title="Available Models")
    table.add_column("Option", style="cyan")
    table.add_column("Model ID", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("RAM", style="yellow")

    for i, model_conf in enumerate(config.AVAILABLE_MODELS):
        table.add_row(str(i + 1), model_conf["id"], model_conf["type"], model_conf["ram"])
    
    utils.console.print(table)
    choice = utils.console.input(f"Enter option [1-{len(config.AVAILABLE_MODELS)}] (default: 1): ")
    
    try:
        index = int(choice) - 1
        if 0 <= index < len(config.AVAILABLE_MODELS):
            return config.AVAILABLE_MODELS[index]
    except (ValueError, IndexError):
        return config.DEFAULT_MODEL_CONFIG # Fallback to default

def load_llm(model_config: dict) -> tuple:
    """
    Main function to load an LLM based on its configuration.
    This acts as a dispatcher.
    """
    model_id = model_config['id']
    model_type = model_config['type']
    
    utils.console.print(f"\n[bold blue]Loading model '{model_id}' of type '{model_type}'...[/bold blue]")

    if not _check_dependencies(model_config):
        raise RuntimeError("Model dependencies are not met. Please install the required packages.")

    # --- Dispatcher ---
    # Call the correct loader based on model type
    if model_type == "causal":
        tokenizer, model = _load_causal_lm(model_id)
    elif model_type == "gptq":
        tokenizer, model = _load_gptq_lm(model_id)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    utils.console.print(f"[green]✓ LLM '{model_id}' loaded successfully.[/green]")
    return tokenizer, model

# The generate_answer function remains the same as your previous version.
def generate_answer(query: str, context: list, tokenizer, llm_model) -> str:
    # ... (Keep your existing generate_answer function here)
    context_text = ' '.join(context)
    if len(context_text) > config.MAX_CONTEXT_CHARS:
        context_text = context_text[:config.MAX_CONTEXT_CHARS] + "..."

    rag_prompt = f"""Answer the following question based *only* on the provided passages. Be concise and extract the answer directly from the text.

Passages:
{context_text}

Question: {query}

Answer:"""

    dialogue_template = [{"role": "user", "content": rag_prompt}]
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = llm_model.generate(
        **input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    answer = tokenizer.decode(outputs[0])
    answer = answer.replace(prompt, '').replace('<bos>', '').replace('<eos>', '').strip()
    return answer