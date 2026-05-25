import sys
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.table import Table
from rag import build_index, get_relevant_chunks
from models import run_all_models
from evaluator import evaluate

console = Console()


def main():
    if len(sys.argv) < 3:
        console.print("[red]Usage: python cli.py <document_path> <question>[/red]")
        sys.exit(1)

    filepath = sys.argv[1]
    question = sys.argv[2]

    console.print("\n[bold]Building index...[/bold]")
    chunks, embeddings = build_index(filepath)

    console.print("[bold]Retrieving relevant chunks...[/bold]")
    relevant_chunks = get_relevant_chunks(question, chunks, embeddings)
    context = "\n\n".join(relevant_chunks)

    console.print("[bold]Running all models in parallel...[/bold]")
    results = run_all_models(question, context)

    console.print("[bold]Evaluating answers...[/bold]\n")
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate, question, context, r["answer"])
            for r in results
        ]
        scores_list = [f.result() for f in futures]

    for result, scores in zip(results, scores_list):
        result["faithfulness"] = scores["faithfulness"]
        result["relevance"] = scores["relevance"]

    table = Table(title=f"Results for: {question}", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Faithfulness", justify="center")
    table.add_column("Relevance", justify="center")
    table.add_column("Latency", justify="center")
    table.add_column("Cost ($)", justify="center")
    table.add_column("Answer", max_width=50)

    for r in results:
        table.add_row(
            r["model"],
            f"{r['faithfulness']}/10",
            f"{r['relevance']}/10",
            f"{r['latency']}s",
            f"{r['cost']:.8f}",
            r["answer"],
        )

    console.print(table)


if __name__ == "__main__":
    main()