import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.orchestrator.workflow import build_evaluator_workflow, GraphState

@pytest.fixture
def mock_acompletion():
    with patch("src.orchestrator.workflow.acompletion") as mock_acomp:
        yield mock_acomp

@pytest.mark.asyncio
async def test_workflow_state_transitions(mock_acompletion):
    # Mock Litellm response to return valid JSON matching ConsistencyEvaluation
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '''
    ```json
    {
      "summary": "Mock summary",
      "issues": [],
      "strengths": ["Good flow"],
      "consistency_score": 9
    }
    ```
    '''
    mock_acompletion.return_value = mock_response

    # Mock scrape_arxiv to avoid network calls and speed up tests
    with patch("src.orchestrator.workflow.scrape_arxiv") as mock_scrape:
        mock_scrape.return_value = "This is a mock paper text. It is very short."
        
        # Also mock asyncio.sleep so we don't wait 4 seconds per chunk in tests!
        with patch("src.orchestrator.workflow.asyncio.sleep", return_value=None):
            app = build_evaluator_workflow()
            
            initial_state = GraphState(
                url="https://arxiv.org/abs/2303.08774",
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
            
            # Verify state transitions
            assert result["arxiv_id"] == "2303.08774"
            assert result["raw_text"] == "This is a mock paper text. It is very short."
            assert len(result["chunks"]) > 0
            assert result["consistency_eval"] is not None
            assert result["consistency_eval"]["consistency_score"] == 9
            assert "Good flow" in result["consistency_eval"]["strengths"]
