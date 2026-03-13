import sys
import os
import asyncio
import shutil
from dotenv import load_dotenv
from rich.console import Console

# Ensure the root directory is accessible for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.workflow import build_evaluator_workflow, GraphState

def main():
    load_dotenv()
    console = Console()
    
    url = "https://arxiv.org/abs/1706.03762"
    console.print(f"[bold blue]Starting BENCHMARK evaluation for:[/bold blue] {url}")
    
    app = build_evaluator_workflow()
    
    async def run_pipeline():
        initial_state = GraphState(
            url=url,
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
            sys.exit(1)
            
        console.print("\n[bold green]Success![/bold green] Analysis complete without errors.")
        
        arxiv_id = result.get("arxiv_id", "unknown")
        pdf_path = f"reports/Judgement_Report_{arxiv_id}.pdf"
        md_path = f"reports/Judgement_Report_{arxiv_id}.md"
        
        os.makedirs("benchmarks", exist_ok=True)
        
        if os.path.exists(pdf_path):
            benchmark_pdf = f"benchmarks/Judgement_Report_1706.03762_Attention_Is_All_You_Need.pdf"
            shutil.copy(pdf_path, benchmark_pdf)
            console.print(f"Benchmark PDF saved to: [cyan]{benchmark_pdf}[/cyan]")
        elif os.path.exists(md_path):
            benchmark_md = f"benchmarks/Judgement_Report_1706.03762_Attention_Is_All_You_Need.md"
            shutil.copy(md_path, benchmark_md)
            console.print(f"PDF not found, Benchmark Markdown saved to: [cyan]{benchmark_md}[/cyan]")

    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main()