import pytest
from unittest.mock import patch, MagicMock
from src.scraper.arxiv_scraper import extract_arxiv_id, scrape_arxiv, ScraperError

def test_extract_arxiv_id():
    assert extract_arxiv_id("https://arxiv.org/abs/2303.08774") == "2303.08774"
    assert extract_arxiv_id("https://arxiv.org/pdf/2303.08774.pdf") == "2303.08774"
    assert extract_arxiv_id("https://arxiv.org/html/2303.08774v2") == "2303.08774"
    assert extract_arxiv_id("2303.08774") == "2303.08774"

@patch('src.scraper.arxiv_scraper.requests.get')
def test_scrape_arxiv_html_success(mock_get, sample_html_content):
    # Mock HTML response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = sample_html_content
    mock_get.return_value = mock_response

    text = scrape_arxiv("2303.08774")
    assert text is not None
    assert "Test Paper" in text
    assert "word" in text
    # Should only call HTML endpoint
    mock_get.assert_called_once_with("https://arxiv.org/html/2303.08774", timeout=10)

@patch('src.scraper.arxiv_scraper.fitz.open')
@patch('src.scraper.arxiv_scraper.requests.get')
def test_scrape_arxiv_pdf_fallback(mock_get, mock_fitz):
    # Mock HTML failure, PDF success
    mock_html_response = MagicMock()
    mock_html_response.status_code = 404
    
    mock_pdf_response = MagicMock()
    mock_pdf_response.status_code = 200
    mock_pdf_response.content = b"%PDF-1.4 mock content"
    
    # Side effect: first call is HTML, second is PDF
    mock_get.side_effect = [mock_html_response, mock_pdf_response]
    
    # Mock PyMuPDF
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Mock PDF Text"
    mock_page.get_images.return_value = []
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz.return_value = mock_doc
    
    text = scrape_arxiv("2303.08774")
    
    assert text is not None
    assert "Mock PDF Text" in text
    assert mock_get.call_count == 2
    mock_fitz.assert_called_once()

@patch('src.scraper.arxiv_scraper.requests.get')
def test_scrape_arxiv_failure(mock_get):
    # Mock both HTML and PDF failing
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response
    
    with pytest.raises(ScraperError) as exc_info:
        scrape_arxiv("invalid_id")
        
    assert "Failed to fetch PDF" in str(exc_info.value)
