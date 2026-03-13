import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

# ==============================================================================
# Agent Output Schemas 
# ==============================================================================
# (Importing the schemas we built in the prompt-template-builder step)
from src.agents.prompt_templates import (
    ConsistencyEvaluation,
    NoveltyEvaluation,
    FactCheckingEvaluation,
    GrammarEvaluation,
    AuthenticityEvaluation
)

# ==============================================================================
# Final Judgement Report Schema
# ==============================================================================

class AgentMetric(BaseModel):
    agent_name: str = Field(..., description="Name of the evaluating agent")
    score: float = Field(..., description="Score given by the agent")
    summary: str = Field(..., description="Brief summary of the agent's findings")
    key_issues: List[str] = Field(default_factory=list, description="Key issues found by the agent")

class FinalJudgementReport(BaseModel):
    paper_title: str = Field(..., description="Title of the evaluated paper")
    arxiv_id: str = Field(..., description="arXiv ID of the evaluated paper")
    executive_summary: str = Field(..., description="A synthesized summary of all agent evaluations")
    
    # Detailed Metrics
    consistency_metrics: ConsistencyEvaluation
    novelty_metrics: NoveltyEvaluation
    fact_checking_metrics: FactCheckingEvaluation
    grammar_metrics: Optional[GrammarEvaluation] = None
    
    # Final Calculated Scores
    overall_score: float = Field(..., description="Overall quality score (0-100)")
    accuracy_score: float = Field(..., description="Calculated accuracy score (0-100)")
    fabrication_risk: float = Field(..., description="Calculated risk of fabrication (percentage 0-100%)")
    
    # Recommendation
    final_verdict: str = Field(..., description="Final recommendation: 'Accept', 'Minor Revisions', 'Major Revisions', or 'Reject'")
    
# ==============================================================================
# Structured Output Extractor Implementation
# ==============================================================================

class ExtractionError(Exception):
    def __init__(self, message: str, raw_output: str, validation_errors: Optional[str] = None):
        super().__init__(message)
        self.raw_output = raw_output
        self.validation_errors = validation_errors

def extract_structured_json(llm_response_text: str, target_model: BaseModel) -> BaseModel:
    """
    Extracts and validates JSON from a raw LLM text response into a Pydantic model.
    Handles common LLM formatting quirks (like markdown code blocks).
    """
    if llm_response_text is None:
        raise ExtractionError(
            message="LLM response content is None. The model failed to generate a response.",
            raw_output=""
        )

    # 1. Clean the output (strip markdown block formatting if present)
    cleaned_text = llm_response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
        
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
        
    cleaned_text = cleaned_text.strip()
    
    # 2. Parse JSON (with repair fallback for malformed LLM output)
    try:
        parsed_json = json.loads(cleaned_text)
    except json.JSONDecodeError as original_error:
        try:
            from json_repair import repair_json
            repaired = repair_json(cleaned_text, return_objects=True)
            if isinstance(repaired, dict) and repaired:
                parsed_json = repaired
            else:
                raise ExtractionError(
                    message=f"Failed to parse valid JSON from LLM response: {str(original_error)}",
                    raw_output=llm_response_text
                )
        except ImportError:
            raise ExtractionError(
                message=f"Failed to parse valid JSON from LLM response: {str(original_error)}",
                raw_output=llm_response_text
            )
        
    # 3. Validate against Pydantic Schema
    try:
        validated_data = target_model(**parsed_json)
        return validated_data
    except ValidationError as e:
        raise ExtractionError(
            message="JSON structure did not match expected schema.",
            raw_output=cleaned_text,
            validation_errors=e.json()
        )

# ==============================================================================
# Report Aggregator
# ==============================================================================

def calculate_fabrication_probability(fact_metrics: FactCheckingEvaluation) -> float:
    """
    Calculates a fabrication probability percentage based on fact-checking risk scores
    and the severity of contradictions found.
    """
    base_risk = float(fact_metrics.fabrication_risk_score)
    
    # Add weight for explicitly contradicted claims
    contradictions = sum(1 for claim in fact_metrics.claims_evaluated if claim.verdict == "contradicted")
    if contradictions > 0:
        base_risk += (contradictions * 20)
        
    return min(100.0, base_risk)

def generate_final_report(
    paper_title: str,
    arxiv_id: str,
    consistency: ConsistencyEvaluation,
    novelty: NoveltyEvaluation,
    fact_checking: FactCheckingEvaluation,
    grammar: Optional[GrammarEvaluation] = None
) -> FinalJudgementReport:
    """
    Aggregates individual agent evaluations into the final Judgement Report structure.
    """
    
    # Calculate Overall Score (Weighted Average)
    # Since novelty_index is qualitative, we use it for summary but don't average it numerically
    # If we need a numerical score for novelty for the 'overall_score', we might have to re-introduce it 
    # or rely on other metrics. For now, let's average consistency and accuracy.
    
    scores = [
        float(consistency.consistency_score),
        float(fact_checking.accuracy_score)
    ]
    # Map grammar rating to score for overall calculation
    if grammar:
        grammar_map = {"High": 100.0, "Medium": 70.0, "Low": 40.0}
        scores.append(grammar_map.get(grammar.grammar_rating, 50.0))
        
    overall_avg = sum(scores) / len(scores)
    
    # Calculate Fabrication Risk
    fab_risk = calculate_fabrication_probability(fact_checking)
    
    # Determine Verdict
    if fab_risk > 50.0 or overall_avg < 40.0:
        verdict = "Reject"
    elif overall_avg < 60.0:
        verdict = "Major Revisions"
    elif overall_avg < 80.0:
        verdict = "Minor Revisions"
    else:
        verdict = "Accept"
        
    # Generate Executive Summary
    grammar_str = f"Grammar is rated as {grammar.grammar_rating}. " if grammar else ""
    exec_summary = (
        f"This paper achieved an overall quality score of {overall_avg:.1f}/100. "
        f"Consistency score: {consistency.consistency_score}/100. "
        f"{grammar_str}"
        f"Novelty Index: {novelty.novelty_index}. "
        f"Accuracy score: {fact_checking.accuracy_score}/100 with a calculated fabrication risk of {fab_risk:.1f}%. "
        f"Based on these metrics, the final recommendation is to {verdict}."
    )

    return FinalJudgementReport(
        paper_title=paper_title,
        arxiv_id=arxiv_id,
        executive_summary=exec_summary,
        consistency_metrics=consistency,
        novelty_metrics=novelty,
        fact_checking_metrics=fact_checking,
        grammar_metrics=grammar,
        overall_score=overall_avg,
        accuracy_score=fact_checking.accuracy_score,
        fabrication_risk=fab_risk,
        final_verdict=verdict
    )
