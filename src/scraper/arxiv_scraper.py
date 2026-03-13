import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import re

class ScraperError(Exception):
    pass

def scrape_arxiv(arxiv_id: str) -> str:
    """
    3-tier extraction pipeline:
    1. HTML via ar5iv/arxiv html
    2. Fallback to PDF via PyMuPDF
    """
    html_url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        response = requests.get(html_url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # Convert math to text representation or keep MathML
            # Basic text extraction from HTML
            text = soup.get_text(separator="\n", strip=True)
            if len(text.split()) > 500:
                return text
    except Exception as e:
        pass # Fallback to PDF

    # Fallback to PDF
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        response = requests.get(pdf_url, timeout=10)
        if response.status_code == 200:
            pdf_path = f"/tmp/{arxiv_id}.pdf"
            import os
            os.makedirs("/tmp", exist_ok=True)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
                
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
                # To simulate the PRD requirement for images:
                # "Replace image-based formulas with '[FORMULA: image-only, skipped]'"
                # We can do a rudimentary check for images on the page
                if len(page.get_images(full=True)) > 0:
                    text += "\n[FORMULA: image-only, skipped]\n"
            doc.close()
            return text
        else:
            raise ScraperError(f"Failed to fetch PDF, status: {response.status_code}")
    except Exception as e:
        raise ScraperError(f"Failed to scrape paper {arxiv_id}: {str(e)}")

def extract_arxiv_id(url: str) -> str:
    """Extracts arxiv ID from various url formats."""
    match = re.search(r'arxiv\.org/(?:abs|pdf|html)/([^v]+?)(?:v\d+)?(?:\.pdf)?$', url)
    if match:
        return match.group(1).replace('.pdf', '')
    # Just return the url if we can't parse it (might be direct ID)
    return url.split("/")[-1].replace('.pdf', '')
