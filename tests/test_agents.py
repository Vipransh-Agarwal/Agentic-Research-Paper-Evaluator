import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from src.orchestrator.workflow import (
    consistency_node, grammar_node, novelty_node, 
    fact_check_node, authenticity_node, GraphState
)

# Helper to create a mock response matching litellm's structure
def create_mock_completion(json_data):
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps(json_data)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response

@pytest.mark.asyncio
@patch('src.orchestrator.workflow.acompletion', new_callable=AsyncMock)
@patch('src.orchestrator.workflow.asyncio.sleep', new_callable=AsyncMock) # Skip sleeps
async def test_consistency_node(mock_sleep, mock_acompletion, mock_llm_response_consistency):
    mock_acompletion.return_value = create_mock_completion(mock_llm_response_consistency)
    
    state = GraphState(chunks=["Chunk 1", "Chunk 2"], arxiv_id="1234")
    result = await consistency_node(state)
    
    assert result["consistency_eval"] is not None
    assert result["consistency_eval"]["consistency_score"] == 90
    assert "Clear structure" in result["consistency_eval"]["strengths"]
    # Ensure it was called twice (once per chunk)
    assert mock_acompletion.call_count == 2

@pytest.mark.asyncio
@patch('src.orchestrator.workflow.acompletion', new_callable=AsyncMock)
@patch('src.orchestrator.workflow.asyncio.sleep', new_callable=AsyncMock)
async def test_grammar_node(mock_sleep, mock_acompletion, mock_llm_response_grammar):
    mock_acompletion.return_value = create_mock_completion(mock_llm_response_grammar)
    
    state = GraphState(chunks=["Chunk 1"], arxiv_id="1234")
    result = await grammar_node(state)
    
    assert result["grammar_eval"] is not None
    assert result["grammar_eval"]["grammar_rating"] == "High"

@pytest.mark.asyncio
@patch('src.orchestrator.workflow.acompletion', new_callable=AsyncMock)
@patch('src.orchestrator.workflow.search_semantic_scholar_impl')
@patch('src.orchestrator.workflow.asyncio.sleep', new_callable=AsyncMock)
async def test_novelty_node(mock_sleep, mock_search, mock_acompletion, mock_llm_response_novelty):
    mock_acompletion.return_value = create_mock_completion(mock_llm_response_novelty)
    mock_search.return_value = [{"title": "Similar Paper"}]
    
    state = GraphState(chunks=["Chunk 1"], arxiv_id="1234")
    result = await novelty_node(state)
    
    assert result["novelty_eval"] is not None
    assert "Breakthrough" in result["novelty_eval"]["novelty_index"]
    assert result["novelty_eval"]["similar_works_referenced"] is True

@pytest.mark.asyncio
@patch('src.orchestrator.workflow.acompletion', new_callable=AsyncMock)
@patch('src.orchestrator.workflow.search_duckduckgo_impl')
@patch('src.orchestrator.workflow.asyncio.sleep', new_callable=AsyncMock)
async def test_fact_check_node(mock_sleep, mock_search, mock_acompletion, mock_llm_response_fact_check):
    # Two calls per chunk: 1. extraction, 2. evaluation
    extraction_response = create_mock_completion("Claim 1")
    eval_response = create_mock_completion(mock_llm_response_fact_check)
    mock_acompletion.side_effect = [extraction_response, eval_response]
    
    mock_search.return_value = [{"title": "Fact check source"}]
    
    state = GraphState(chunks=["Chunk 1"], arxiv_id="1234")
    result = await fact_check_node(state)
    
    assert result["fact_check_eval"] is not None
    assert result["fact_check_eval"]["accuracy_score"] == 95
    assert result["fact_check_eval"]["fabrication_risk_score"] == 5

@pytest.mark.asyncio
@patch('src.orchestrator.workflow.acompletion', new_callable=AsyncMock)
@patch('src.orchestrator.workflow.asyncio.sleep', new_callable=AsyncMock)
async def test_authenticity_node(mock_sleep, mock_acompletion, mock_llm_response_authenticity):
    mock_acompletion.return_value = create_mock_completion(mock_llm_response_authenticity)
    
    state = GraphState(
        consistency_eval={"summary": "C"},
        grammar_eval={"summary": "G"},
        novelty_eval={"summary": "N"},
        fact_check_eval={"summary": "F"}
    )
    result = await authenticity_node(state)
    
    assert result["authenticity_eval"] is not None
    assert result["authenticity_eval"]["fabrication_probability"] == 5.0
    assert result["authenticity_eval"]["metrics"]["claimVerificationRatio"] == 95.0
