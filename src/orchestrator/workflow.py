import asyncio
import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import json
from datetime import datetime, timezone

from src.scraper.arxiv_scraper import scrape_arxiv, extract_arxiv_id
from src.scraper.tools import search_semantic_scholar_impl, SearchSemanticScholarParams, search_duckduckgo_impl, SearchDuckDuckGoParams
from src.processing.chunker import PaperChunker, LLMCacheManager, generate_hash
from src.agents.prompt_templates import (
    build_consistency_prompt, ConsistencyEvaluation,
    build_grammar_prompt, GrammarEvaluation,
    build_novelty_prompt, NoveltyEvaluation,
    build_fact_checking_prompt, FactCheckingEvaluation,
    build_authenticity_prompt, AuthenticityEvaluation
)
from src.output.extractor import extract_structured_json, generate_final_report
from litellm import acompletion

# Global flags to ensure status is logged once
_gemini_status_logged = False
_or_status_logged = False

async def smart_acompletion(model: str, messages: List[Dict[str, str]], fallbacks: List[str] = None, **kwargs):
    global _gemini_status_logged, _or_status_logged
    
    fallbacks = fallbacks or []
    all_models = [model] + fallbacks
    valid_models = []
    
    for i, m in enumerate(all_models):
        is_or = "openrouter" in m.lower()
        is_gemini = "gemini" in m.lower() and not is_or

        if is_gemini:
            key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                if not _gemini_status_logged:
                    next_model_name = "open router" if any("openrouter" in next_m.lower() for next_m in all_models[i+1:]) else "next available model"
                    logger.info(f"gemini key not found, falling back to {next_model_name}")
                    _gemini_status_logged = True
                continue
            else:
                if not _gemini_status_logged:
                    logger.info("gemini key found")
                    logger.info("gemini model found")
                    _gemini_status_logged = True
        
        elif is_or:
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                if not _or_status_logged:
                    next_model_name = "ollama" if any("ollama" in next_m.lower() for next_m in all_models[i+1:]) else "next available model"
                    logger.info(f"openrouter key not found, falling back to {next_model_name}")
                    _or_status_logged = True
                continue
            else:
                if not _or_status_logged:
                    logger.info("openrouter key found")
                    logger.info("open router model found")
                    _or_status_logged = True
        
        valid_models.append(m)

    if not valid_models:
        # If no models have keys, just try the original call and let litellm raise its error
        return await acompletion(model=model, messages=messages, fallbacks=fallbacks, **kwargs)

    return await acompletion(model=valid_models[0], messages=messages, fallbacks=valid_models[1:] if len(valid_models) > 1 else None, **kwargs)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# LLM Constants
try:
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", 16000))
except (ValueError, TypeError):
    MAX_OUTPUT_TOKENS = 16000

# Initialize singletons
chunker = PaperChunker()
cache_manager = LLMCacheManager()

# Helper for UTC time
def get_utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# Define the State for the LangGraph Workflow
class GraphState(TypedDict):
    url: str
    arxiv_id: str
    raw_text: Optional[str]
    chunks: List[str]  # We'll store the text of the chunks here for simplicity
    
    # Agent evaluations
    consistency_eval: Optional[Dict[str, Any]]
    grammar_eval: Optional[Dict[str, Any]]
    novelty_eval: Optional[Dict[str, Any]]
    fact_check_eval: Optional[Dict[str, Any]]
    authenticity_eval: Optional[Dict[str, Any]]
    
    final_report: Optional[str]
    final_verdict: Optional[str]
    overall_score: Optional[float]
    errors: List[str]

# ---------------------------------------------------------
# Node 1: Scrape
# ---------------------------------------------------------
def scrape_node(state: GraphState):
    url = state["url"]
    logger.info(f"--- Node: Scrape --- | URL: {url}")
    arxiv_id = extract_arxiv_id(url)
    try:
        raw_text = scrape_arxiv(arxiv_id)
        logger.info(f"Scraped {len(raw_text)} characters.")
        return {"raw_text": raw_text, "arxiv_id": arxiv_id}
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return {"errors": [f"Scrape error: {str(e)}"], "arxiv_id": arxiv_id}

# ---------------------------------------------------------
# Node 2: Decompose & Chunk
# ---------------------------------------------------------
def decompose_node(state: GraphState):
    logger.info("--- Node: Decompose --- | Chunking strictly under 16k tokens")
    if not state.get("raw_text"):
        return {"chunks": []}
        
    text_chunks = chunker.chunk_text(state["raw_text"], state.get("arxiv_id", "unknown"))
    chunks_text = [c.text for c in text_chunks]
    logger.info(f"Created {len(chunks_text)} chunks.")
    return {"chunks": chunks_text}

# ---------------------------------------------------------
# Analyze Nodes: 5 Specialized Agents
# ---------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=10))
async def consistency_node(state: GraphState):
    logger.info("--- Node: Consistency Agent ---")
    chunks = state.get("chunks", [])
    if not chunks:
        return {"consistency_eval": None}
        
    all_issues = []
    all_strengths = []
    scores = []
    summaries = []
    
    primary_model = os.getenv("LLM_MODEL", "gemini/gemini-3.1-flash-lite-preview")
    fallback_models = ["openrouter/google/gemini-3.1-flash-lite-preview", "ollama/llama3"]
    current_time = get_utc_now()
    
    for idx, chunk in enumerate(chunks):
        if idx > 0:
            await asyncio.sleep(4)
        
        prompt_vars = {
            "paper_title": f"Paper {state.get('arxiv_id')}",
            "paper_abstract": "Abstract extracted from text...",
            "paper_text": chunk,
            "focus_area": "Logical flow and structural integrity",
            "current_utc_time": current_time
        }
        
        prompt = build_consistency_prompt(prompt_vars)
        messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
        
        chunk_hash = generate_hash(chunk)
        cached = cache_manager.get_cached_response(chunk_hash, "consistency")
        
        if cached:
            logger.info(f"Consistency Agent: Using cached response for chunk {idx+1}/{len(chunks)}.")
            eval_dict = cached
        else:
            logger.info(f"Consistency Agent: Calling LLM for chunk {idx+1}/{len(chunks)}...")
            try:
                response = await smart_acompletion(model=primary_model, messages=messages, temperature=0.2, fallbacks=fallback_models, max_tokens=MAX_OUTPUT_TOKENS)
                content = response.choices[0].message.content
                if content is None:
                    logger.warning(f"Consistency Agent: LLM returned None content for chunk {idx+1}/{len(chunks)}, skipping chunk.")
                    continue
                eval_result = extract_structured_json(content, ConsistencyEvaluation)
                eval_dict = eval_result.model_dump()
                cache_manager.save_response(chunk_hash, "consistency", eval_dict)
            except Exception as e:
                logger.error(f"Consistency Agent failed on chunk {idx+1}: {e}")
                return {"errors": state.get("errors", []) + [f"Consistency Agent Error: {str(e)}"]}
        
        if eval_dict.get("issues"):
            all_issues.extend(eval_dict["issues"])
        if eval_dict.get("strengths"):
            all_strengths.extend(eval_dict["strengths"])
        if eval_dict.get("consistency_score") is not None:
            scores.append(eval_dict["consistency_score"])
        if eval_dict.get("summary"):
            summaries.append(eval_dict["summary"])

    avg_score = int(sum(scores) / len(scores)) if scores else 0
    final_summary = " ".join(summaries)
    
    final_eval = {
        "summary": final_summary[:500] + "..." if len(final_summary) > 500 else final_summary,
        "issues": all_issues,
        "strengths": list(set(all_strengths)),
        "consistency_score": avg_score
    }
    return {"consistency_eval": final_eval}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=10))
async def grammar_node(state: GraphState):
    logger.info("--- Node: Grammar Agent ---")
    chunks = state.get("chunks", [])
    if not chunks:
        return {"grammar_eval": None}
        
    all_issues = []
    ratings = []
    summaries = []
    
    primary_model = os.getenv("LLM_MODEL", "gemini/gemini-3.1-flash-lite-preview")
    fallback_models = ["openrouter/google/gemini-3.1-flash-lite-preview", "ollama/llama3"]
    current_time = get_utc_now()
    
    for idx, chunk in enumerate(chunks):
        if idx > 0:
            await asyncio.sleep(4)
        
        prompt_vars = {
            "paper_text": chunk,
            "current_utc_time": current_time
        }
        prompt = build_grammar_prompt(prompt_vars)
        messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
        
        chunk_hash = generate_hash(chunk)
        cached = cache_manager.get_cached_response(chunk_hash, "grammar")
        
        if cached:
            logger.info(f"Grammar Agent: Using cached response for chunk {idx+1}/{len(chunks)}.")
            eval_dict = cached
        else:
            logger.info(f"Grammar Agent: Calling LLM for chunk {idx+1}/{len(chunks)}...")
            try:
                response = await smart_acompletion(model=primary_model, messages=messages, temperature=0.2, fallbacks=fallback_models, max_tokens=MAX_OUTPUT_TOKENS)
                content = response.choices[0].message.content
                if content is None:
                    logger.warning(f"Grammar Agent: LLM returned None content for chunk {idx+1}/{len(chunks)}, skipping chunk.")
                    continue
                eval_result = extract_structured_json(content, GrammarEvaluation)
                eval_dict = eval_result.model_dump()
                cache_manager.save_response(chunk_hash, "grammar", eval_dict)
            except Exception as e:
                logger.error(f"Grammar Agent failed on chunk {idx+1}: {e}")
                return {"errors": state.get("errors", []) + [f"Grammar Agent Error: {str(e)}"]}
        
        if eval_dict.get("issues"):
            all_issues.extend(eval_dict["issues"])
        if eval_dict.get("grammar_rating"):
            ratings.append(eval_dict["grammar_rating"])
        if eval_dict.get("summary"):
            summaries.append(eval_dict["summary"])

    # Aggregate rating: Take the "lowest" rating found
    rating_rank = {"High": 3, "Medium": 2, "Low": 1}
    min_rank = min([rating_rank.get(r, 2) for r in ratings]) if ratings else 2
    final_rating = [k for k, v in rating_rank.items() if v == min_rank][0]
    
    final_summary = " ".join(summaries)
    
    final_eval = {
        "summary": final_summary[:500] + "..." if len(final_summary) > 500 else final_summary,
        "issues": all_issues,
        "grammar_rating": final_rating
    }
    return {"grammar_eval": final_eval}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=10))
async def novelty_node(state: GraphState):
    logger.info("--- Node: Novelty Agent ---")
    chunks = state.get("chunks", [])
    if not chunks:
        return {"novelty_eval": None}
        
    all_findings = []
    indices = []
    summaries = []
    similar_works = []
    
    primary_model = os.getenv("LLM_MODEL", "gemini/gemini-3.1-flash-lite-preview")
    fallback_models = ["openrouter/google/gemini-3.1-flash-lite-preview", "ollama/llama3"]
    current_time = get_utc_now()
    
    # Pre-fetch context from Semantic Scholar
    arxiv_id = state.get('arxiv_id', '')
    query = f"arxiv {arxiv_id}"
    logger.info(f"Novelty Agent: Querying Semantic Scholar for '{query}'...")
    search_params = SearchSemanticScholarParams(query=query, limit=3)
    search_results = search_semantic_scholar_impl(search_params)
    injected_context = f"Semantic Scholar Findings:\n{json.dumps(search_results)}\n\n"
    
    for idx, chunk in enumerate(chunks):
        if idx > 0:
            await asyncio.sleep(4)
        
        prompt_vars = {
            "paper_title": f"Paper {arxiv_id}",
            "paper_abstract": "Abstract context",
            "paper_text": injected_context + chunk,
            "domain_knowledge_context": "General scientific baseline",
            "current_utc_time": current_time
        }
        prompt = build_novelty_prompt(prompt_vars)
        messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
        
        chunk_hash = generate_hash(injected_context + chunk)
        cached = cache_manager.get_cached_response(chunk_hash, "novelty")
        
        if cached:
            logger.info(f"Novelty Agent: Using cached response for chunk {idx+1}/{len(chunks)}.")
            eval_dict = cached
        else:
            logger.info(f"Novelty Agent: Calling LLM for chunk {idx+1}/{len(chunks)}...")
            try:
                response = await smart_acompletion(model=primary_model, messages=messages, temperature=0.2, fallbacks=fallback_models, max_tokens=MAX_OUTPUT_TOKENS)
                content = response.choices[0].message.content
                if content is None:
                    logger.warning(f"Novelty Agent: LLM returned None content for chunk {idx+1}/{len(chunks)}, skipping chunk.")
                    continue
                eval_result = extract_structured_json(content, NoveltyEvaluation)
                eval_dict = eval_result.model_dump()
                cache_manager.save_response(chunk_hash, "novelty", eval_dict)
            except Exception as e:
                logger.error(f"Novelty Agent failed on chunk {idx+1}: {e}")
                return {"errors": state.get("errors", []) + [f"Novelty Agent Error: {str(e)}"]}
        
        if eval_dict.get("findings"):
            all_findings.extend(eval_dict["findings"])
        if eval_dict.get("novelty_index"):
            indices.append(eval_dict["novelty_index"])
        if eval_dict.get("summary"):
            summaries.append(eval_dict["summary"])
        similar_works.append(eval_dict.get("similar_works_referenced", False))

    final_index = "; ".join(list(set(indices)))
    final_summary = " ".join(summaries)
    
    final_eval = {
        "summary": final_summary[:500] + "..." if len(final_summary) > 500 else final_summary,
        "findings": all_findings,
        "similar_works_referenced": any(similar_works),
        "novelty_index": final_index
    }
    return {"novelty_eval": final_eval}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=10))
async def fact_check_node(state: GraphState):
    logger.info("--- Node: Fact-Checking Agent ---")
    chunks = state.get("chunks", [])
    if not chunks:
        return {"fact_check_eval": None}
        
    all_claims = []
    risk_scores = []
    accuracy_scores = []
    summaries = []
    
    primary_model = os.getenv("LLM_MODEL", "gemini/gemini-3.1-flash-lite-preview")
    fallback_models = ["openrouter/google/gemini-3.1-flash-lite-preview", "ollama/llama3"]
    arxiv_id = state.get('arxiv_id', '')
    current_time = get_utc_now()
    
    for idx, chunk in enumerate(chunks):
        if idx > 0:
            await asyncio.sleep(4)
            
        chunk_hash = generate_hash(chunk)
        cached = cache_manager.get_cached_response(chunk_hash, "fact_check")
        
        if cached:
            logger.info(f"Fact-Checking Agent: Using cached response for chunk {idx+1}/{len(chunks)}.")
            eval_dict = cached
        else:
            logger.info(f"Fact-Checking Agent: Extracting claims and querying DDG for chunk {idx+1}/{len(chunks)}...")
            
            # 1. Quick claim extraction
            extract_prompt = f"Extract 1-3 key verifiable claims from the following text as a short comma-separated list. If no specific factual claims, return 'None'. Text:\n{chunk[:2000]}"
            try:
                claim_response = await smart_acompletion(model=primary_model, messages=[{"role": "user", "content": extract_prompt}], temperature=0.0, fallbacks=fallback_models, max_tokens=MAX_OUTPUT_TOKENS)
                raw_content = claim_response.choices[0].message.content
                claims_text = raw_content.strip() if raw_content else "None"
                
                injected_context = ""
                if claims_text.lower() != 'none':
                    query = f"arxiv {arxiv_id} {claims_text[:100]}"
                    search_params = SearchDuckDuckGoParams(query=query, max_results=3)
                    search_results = search_duckduckgo_impl(search_params)
                    injected_context = f"DuckDuckGo Findings for ({claims_text[:100]}):\n{json.dumps(search_results)}\n\n"
                    
                await asyncio.sleep(4) # Pace before main evaluation
            except Exception as e:
                injected_context = f"Search failed: {str(e)}\n\n"

            # 2. Main fact check prompt
            prompt_vars = {
                "paper_text": injected_context + chunk,
                "extract_count": 5,
                "external_knowledge_allowed": True,
                "current_utc_time": current_time
            }
            prompt = build_fact_checking_prompt(prompt_vars)
            messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
            
            try:
                response = await smart_acompletion(model=primary_model, messages=messages, temperature=0.0, fallbacks=fallback_models, max_tokens=MAX_OUTPUT_TOKENS)
                content = response.choices[0].message.content
                if content is None:
                    logger.warning(f"Fact-Checking Agent: LLM returned None content for chunk {idx+1}/{len(chunks)}, skipping chunk.")
                    continue
                eval_result = extract_structured_json(content, FactCheckingEvaluation)
                eval_dict = eval_result.model_dump()
                cache_manager.save_response(chunk_hash, "fact_check", eval_dict)
            except Exception as e:
                logger.error(f"Fact-Checking Agent failed on chunk {idx+1}: {e}")
                return {"errors": state.get("errors", []) + [f"Fact-Checking Agent Error: {str(e)}"]}
        
        if eval_dict.get("claims_evaluated"):
            all_claims.extend(eval_dict["claims_evaluated"])
        if eval_dict.get("fabrication_risk_score") is not None:
            risk_scores.append(eval_dict["fabrication_risk_score"])
        if eval_dict.get("accuracy_score") is not None:
            accuracy_scores.append(eval_dict["accuracy_score"])
        if eval_dict.get("summary"):
            summaries.append(eval_dict["summary"])

    avg_risk = int(sum(risk_scores) / len(risk_scores)) if risk_scores else 5
    avg_accuracy = int(sum(accuracy_scores) / len(accuracy_scores)) if accuracy_scores else 95
    final_summary = " ".join(summaries)
    
    final_eval = {
        "summary": final_summary[:500] + "..." if len(final_summary) > 500 else final_summary,
        "claims_evaluated": all_claims,
        "fabrication_risk_score": avg_risk,
        "accuracy_score": avg_accuracy
    }
    return {"fact_check_eval": final_eval}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=10))
async def authenticity_node(state: GraphState):
    logger.info("--- Node: Authenticity Agent (Synthesizer) ---")
    await asyncio.sleep(4)
    
    consistency_summary = state.get("consistency_eval", {}).get("summary", "None") if state.get("consistency_eval") else "None"
    grammar_summary = state.get("grammar_eval", {}).get("summary", "None") if state.get("grammar_eval") else "None"
    novelty_summary = state.get("novelty_eval", {}).get("summary", "None") if state.get("novelty_eval") else "None"
    fact_check_summary = state.get("fact_check_eval", {}).get("summary", "None") if state.get("fact_check_eval") else "None"
    current_time = get_utc_now()
    
    prompt_vars = {
        "consistency_summary": consistency_summary,
        "grammar_summary": grammar_summary,
        "novelty_summary": novelty_summary,
        "fact_check_summary": fact_check_summary,
        "current_utc_time": current_time
    }
    
    prompt = build_authenticity_prompt(prompt_vars)
    messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
    
    primary_model = os.getenv("LLM_MODEL", "gemini/gemini-3.1-flash-lite-preview")
    fallback_models = ["openrouter/google/gemini-3.1-flash-lite-preview", "ollama/llama3"]
    
    try:
        response = await smart_acompletion(model=primary_model, messages=messages, temperature=0.2, fallbacks=fallback_models, max_tokens=MAX_OUTPUT_TOKENS)
        content = response.choices[0].message.content
        if content is None:
            logger.warning("Authenticity Agent: LLM returned None content, skipping.")
            return {"authenticity_eval": None}
        logger.info(f"Authenticity Agent raw response: {content[:500]}")
        eval_result = extract_structured_json(content, AuthenticityEvaluation)
        eval_dict = eval_result.model_dump()
        return {"authenticity_eval": eval_dict}
    except Exception as e:
        logger.error(f"Authenticity Agent failed: {e}")
        return {"errors": state.get("errors", []) + [f"Authenticity Agent Error: {str(e)}"]}


# ---------------------------------------------------------
# Node: Report Generator
# ---------------------------------------------------------
def report_node(state: GraphState):
    logger.info("--- Node: Report Generator ---")
    arxiv_id = state.get("arxiv_id", "unknown")
    
    try:
        consistency_eval = ConsistencyEvaluation(**state.get("consistency_eval")) if state.get("consistency_eval") else None
        novelty_eval = NoveltyEvaluation(**state.get("novelty_eval")) if state.get("novelty_eval") else None
        fact_eval = FactCheckingEvaluation(**state.get("fact_check_eval")) if state.get("fact_check_eval") else None
        grammar_eval = GrammarEvaluation(**state.get("grammar_eval")) if state.get("grammar_eval") else None
        
        if not consistency_eval or not novelty_eval or not fact_eval:
            return {"final_report": "Error: Missing required agent evaluations."}
            
        final_report = generate_final_report(
            paper_title=f"ArXiv Paper {arxiv_id}",
            arxiv_id=arxiv_id,
            consistency=consistency_eval,
            novelty=novelty_eval,
            fact_checking=fact_eval,
            grammar=grammar_eval
        )
        
        # Format as Markdown according to NEW REQUIREMENTS
        md_content = f"# Judgement Report: {final_report.paper_title}\n\n"
        md_content += f"**Verdict:** {final_report.final_verdict}\n\n"
        md_content += f"**Overall Quality Score:** {final_report.overall_score:.1f}/100\n\n"
        
        md_content += f"## Executive Summary\n{final_report.executive_summary}\n\n"
        
        md_content += f"## Detailed Scores\n"
        md_content += f"- **Consistency Score:** {final_report.consistency_metrics.consistency_score}/100\n"
        if grammar_eval:
            md_content += f"- **Grammar Rating:** {final_report.grammar_metrics.grammar_rating}\n"
        md_content += f"- **Novelty Index:** {final_report.novelty_metrics.novelty_index}\n"
        
        md_content += f"\n## Fact Check Log\n"
        for claim in final_report.fact_checking_metrics.claims_evaluated:
            status_icon = "✅" if claim.verdict == "supported" else "❌" if claim.verdict == "contradicted" else "⚠️"
            md_content += f"- {status_icon} **{claim.claim}**\n"
            md_content += f"  - *Verdict:* {claim.verdict}\n"
            md_content += f"  - *Evidence:* {claim.evidence}\n"
            
        md_content += f"\n**Accuracy Score:** {final_report.accuracy_score:.1f}/100\n"
        md_content += f"**Fabrication Risk:** {final_report.fabrication_risk:.1f}%\n"
            
        # Write to disk
        os.makedirs("reports", exist_ok=True)
        report_path = f"reports/Judgement_Report_{arxiv_id}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        logger.info(f"Report written to {report_path}")
        
        # WeasyPrint PDF Generation
        try:
            import markdown
            from weasyprint import HTML
            html_out = markdown.markdown(md_content)
            pdf_path = f"reports/Judgement_Report_{arxiv_id}.pdf"
            HTML(string=html_out).write_pdf(pdf_path)
            logger.info(f"PDF Report written to {pdf_path}")
        except Exception as e:
            logger.warning(f"Could not generate PDF (WeasyPrint may be missing dependencies): {e}")
            
        return {
            "final_report": md_content,
            "final_verdict": final_report.final_verdict,
            "overall_score": final_report.overall_score
        }
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {"errors": state.get("errors", []) + [f"Report Gen Error: {str(e)}"]}

# ---------------------------------------------------------
# Build LangGraph Workflow
# ---------------------------------------------------------
def build_evaluator_workflow():
    workflow = StateGraph(GraphState)
    
    # 1. Add Nodes
    workflow.add_node("scrape", scrape_node)
    workflow.add_node("decompose", decompose_node)
    workflow.add_node("consistency", consistency_node)
    workflow.add_node("grammar", grammar_node)
    workflow.add_node("novelty", novelty_node)
    workflow.add_node("fact_checker", fact_check_node)
    workflow.add_node("authenticity", authenticity_node)
    workflow.add_node("report", report_node)
    
    # 2. Add Edges (Sequential Flow to naturally govern pacing)
    workflow.set_entry_point("scrape")
    workflow.add_edge("scrape", "decompose")
    workflow.add_edge("decompose", "consistency")
    workflow.add_edge("consistency", "grammar")
    workflow.add_edge("grammar", "novelty")
    workflow.add_edge("novelty", "fact_checker")
    workflow.add_edge("fact_checker", "authenticity")
    workflow.add_edge("authenticity", "report")
    workflow.add_edge("report", END)
    
    # Compile
    return workflow.compile()

# Example local execution entry point
if __name__ == "__main__":
    app = build_evaluator_workflow()
    
    # Run async pipeline
    async def run_pipeline():
        initial_state = GraphState(
            url="https://arxiv.org/abs/test",
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
        
        logger.info("Starting Agentic Evaluation Pipeline...")
        result = await app.ainvoke(initial_state)
        logger.info("Pipeline Complete. Final Report Generated.")
        
    asyncio.run(run_pipeline())
