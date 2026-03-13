import pytest
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock
from src.orchestrator.workflow import build_evaluator_workflow, GraphState

def create_mock_completion(json_data):
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps(json_data) if isinstance(json_data, dict) else json_data
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response

@pytest.mark.asyncio
@patch('src.orchestrator.workflow.scrape_arxiv')
@patch('src.orchestrator.workflow.acompletion', new_callable=AsyncMock)
@patch('src.orchestrator.workflow.search_semantic_scholar_impl')
@patch('src.orchestrator.workflow.search_duckduckgo_impl')
@patch('src.orchestrator.workflow.asyncio.sleep', new_callable=AsyncMock)
async def test_end_to_end_workflow(
    mock_sleep, mock_ddg, mock_ss, mock_acompletion, mock_scrape,
    mock_llm_response_consistency, mock_llm_response_grammar,
    mock_llm_response_novelty, mock_llm_response_fact_check,
    mock_llm_response_authenticity
):
    # Mock Scraper
    mock_scrape.return_value = "This is a test paper that will be evaluated end-to-end. " * 50
    
    # Mock Searches
    mock_ss.return_value = []
    mock_ddg.return_value = []
    
    # Mock LLM calls in the order they are called in the graph (1 chunk)
    # Order: Consistency -> Grammar -> Novelty -> FactCheck (extract, eval) -> Authenticity
    mock_acompletion.side_effect = [
        create_mock_completion(mock_llm_response_consistency),
        create_mock_completion(mock_llm_response_grammar),
        create_mock_completion(mock_llm_response_novelty),
        create_mock_completion("Claim 1"), # Fact check extraction
        create_mock_completion(mock_llm_response_fact_check),
        create_mock_completion(mock_llm_response_authenticity),
    ]

    app = build_evaluator_workflow()
    
    initial_state = GraphState(
        url="https://arxiv.org/abs/1234.5678",
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
    
    result = await app.ainvoke(initial_state)
    
    # Verify no errors
    assert len(result.get("errors", [])) == 0
    
    # Verify evaluations are populated
    assert result["consistency_eval"] is not None
    assert result["grammar_eval"] is not None
    assert result["novelty_eval"] is not None
    assert result["fact_check_eval"] is not None
    assert result["authenticity_eval"] is not None
    
    # Verify report is generated
    assert result["final_report"] is not None
    assert "Judgement Report" in result["final_report"]
    
    # Verify files were written
    report_path = f"reports/Judgement_Report_1234.5678.md"
    assert os.path.exists(report_path)
    
    # Cleanup generated files
    if os.path.exists(report_path):
        os.remove(report_path)
    pdf_path = f"reports/Judgement_Report_1234.5678.pdf"
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
