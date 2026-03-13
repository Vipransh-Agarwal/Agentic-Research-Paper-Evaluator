import argparse
import sys
import os
import asyncio
import json
from dotenv import load_dotenv
from rich.console import Console

# Ensure the root directory is accessible for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.workflow import build_evaluator_workflow, GraphState

def main():
    load_dotenv()
    console = Console()
    
    parser = argparse.ArgumentParser(description="Agentic Research Paper Evaluator")
    parser.add_argument("--url", type=str, required=True, help="The arXiv URL to evaluate (e.g., https://arxiv.org/abs/xxxx.xxxxx)")
    
    args = parser.parse_args()
    
    console.print(f"[bold blue]Starting evaluation for:[/bold blue] {args.url}")
    console.print("[dim]Initializing LangGraph orchestrator...[/dim]")

    app = build_evaluator_workflow()
    
    async def run_pipeline():
        initial_state = GraphState(
            url=args.url,
            arxiv_id="",
            raw_text=None,
            chunks=[],
            consistency_eval=None,
            grammar_eval=None,
            novelty_eval=None,
            fact_check_eval=None,
            authenticity_eval=None,
            final_report=None,
            errors=[]
        )
        
        console.print("\n[bold]--- Pipeline Execution Started ---[/bold]")
        result = await app.ainvoke(initial_state)
        console.print("\n[bold]--- Pipeline Execution Complete ---[/bold]")
        
        if result.get("errors") and len(result["errors"]) > 0:
            console.print("\n[bold red]Pipeline encountered errors:[/bold red]")
            for err in result["errors"]:
                console.print(f"[bold red]- {err}[/bold red]")
            console.print("\n[bold yellow]Exiting due to errors. Report generation failed or incomplete.[/bold yellow]")
            sys.exit(1)
            
        console.print("\n[bold green]Success![/bold green] Analysis complete without errors.")
        
        # Determine paths
        arxiv_id = result.get("arxiv_id", "unknown")
        md_path = f"reports/Judgement_Report_{arxiv_id}.md"
        pdf_path = f"reports/Judgement_Report_{arxiv_id}.pdf"
        
        if os.path.exists(pdf_path):
            console.print(f"Generated PDF Report: [cyan]{pdf_path}[/cyan]")
        elif os.path.exists(md_path):
            console.print(f"Generated Markdown Report: [cyan]{md_path}[/cyan]")
            
        console.print("\n[dim]--- Intermediate State Dump (Metadata) ---[/dim]")
        # Print relevant state (excluding full chunks/raw_text to avoid clutter)
        dump_state = {
            "url": result.get("url"),
            "arxiv_id": result.get("arxiv_id"),
            "chunk_count": len(result.get("chunks", [])),
            "consistency_score": result.get("consistency_eval", {}).get("consistency_score") if result.get("consistency_eval") else None,
            "novelty_score": result.get("novelty_eval", {}).get("novelty_score") if result.get("novelty_eval") else None,
            "fact_check_score": result.get("fact_check_eval", {}).get("fact_score") if result.get("fact_check_eval") else None,
            "fabrication_probability": result.get("authenticity_eval", {}).get("fabrication_probability") if result.get("authenticity_eval") else None
        }
        console.print_json(data=dump_state)

    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main()

