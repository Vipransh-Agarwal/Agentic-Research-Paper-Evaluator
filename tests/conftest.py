import pytest
import os
import json
from unittest.mock import patch

@pytest.fixture
def sample_html_content():
    return """
    <html>
        <body>
            <div class="ltx_document">
                <h1>Test Paper</h1>
                <p>This is a test abstract. It has more than a few words to simulate a real paper.</p>
                <p>""" + " word" * 501 + """</p>
            </div>
        </body>
    </html>
    """

@pytest.fixture
def mock_llm_response_consistency():
    return {
        "summary": "The paper is consistent.",
        "issues": [],
        "strengths": ["Clear structure"],
        "consistency_score": 9
    }

@pytest.fixture
def mock_llm_response_grammar():
    return {
        "summary": "Good grammar.",
        "issues": [],
        "grammar_score": 8
    }

@pytest.fixture
def mock_llm_response_novelty():
    return {
        "summary": "Very novel.",
        "findings": [{"aspect": "Methodology", "novelty_level": "high", "justification": "New method proposed"}],
        "similar_works_referenced": True,
        "novelty_score": 8
    }

@pytest.fixture
def mock_llm_response_fact_check():
    return {
        "summary": "Facts seem correct.",
        "claims_evaluated": [{"claim": "X=Y", "verdict": "supported", "evidence": "Because math.", "confidence": "high"}],
        "fabrication_risk_score": 2,
        "fact_score": 9
    }

@pytest.fixture
def mock_llm_response_authenticity():
    return {
        "summary": "Paper seems authentic.",
        "fabrication_probability": 5.0,
        "metrics": {
            "claimVerificationRatio": 95.0,
            "logicalDisconnectPenalty": 5.0,
            "citationIntegrityIndex": 90.0,
            "methodologicalVaguenessScore": 10.0
        }
    }

@pytest.fixture(autouse=True)
def setup_env():
    # Set mock environment variables for testing
    os.environ["LLM_MODEL"] = "test-model"
    os.environ["GEMINI_API_KEY"] = "test-key"
    yield
    # Cleanup if needed

@pytest.fixture(autouse=True)
def mock_cache_manager():
    with patch('src.orchestrator.workflow.cache_manager') as mock_cache:
        mock_cache.get_cached_response.return_value = None
        yield mock_cache
