import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError

# ==============================================================================
# Pydantic Schemas for Validation
# ==============================================================================

class SearchArxivParams(BaseModel):
    query: str = Field(..., description="The search query for finding research papers on arXiv (e.g., 'au:Einstein AND ti:relativity').")
    max_results: int = Field(5, ge=1, le=50, description="Maximum number of papers to return.")
    sort_by: str = Field("relevance", description="Criteria to sort by. Options: 'relevance', 'lastUpdatedDate', 'submittedDate'.")
    sort_order: str = Field("descending", description="Order of sorting. Options: 'ascending', 'descending'.")

class ScrapeArxivPaperParams(BaseModel):
    arxiv_id: str = Field(..., description="The unique arXiv identifier of the paper (e.g., '2303.08774').")
    extract_figures: bool = Field(False, description="Whether to attempt extracting figures/tables (advanced).")
    extract_formulas: bool = Field(True, description="Whether to extract MathML/LaTeX formulas.")

class SearchSemanticScholarParams(BaseModel):
    query: str = Field(..., description="The search query for finding related research papers.")
    limit: int = Field(3, ge=1, le=10, description="Maximum number of papers to return.")

class SearchDuckDuckGoParams(BaseModel):
    query: str = Field(..., description="The search query for finding general web information.")
    max_results: int = Field(3, ge=1, le=10, description="Maximum number of web results to return.")

# ==============================================================================
# Tool JSON Schemas (OpenAI/LiteLLM Format)
# ==============================================================================

SEARCH_SEMANTIC_SCHOLAR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_semantic_scholar",
        "description": "Search the Semantic Scholar database for related academic papers. Useful for checking novelty.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "limit": {"type": "integer", "description": "Maximum number of results to return.", "default": 3}
            },
            "required": ["query"]
        }
    }
}

SEARCH_DUCKDUCKGO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_duckduckgo",
        "description": "Search the web using DuckDuckGo. Useful for fact-checking general claims.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {"type": "integer", "description": "Maximum number of results.", "default": 3}
            },
            "required": ["query"]
        }
    }
}

SEARCH_ARXIV_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_arxiv",
        "description": "Search the arXiv database for research papers. Returns titles, authors, IDs, and abstracts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for finding research papers on arXiv."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "default": "relevance"
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["ascending", "descending"],
                    "default": "descending"
                }
            },
            "required": ["query"]
        }
    }
}

SCRAPE_ARXIV_PAPER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "scrape_arxiv_paper",
        "description": "Fetches and extracts the full text and metadata of a specific arXiv paper using its ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "The unique arXiv identifier of the paper (e.g., '2303.08774')."
                },
                "extract_figures": {
                    "type": "boolean",
                    "description": "Attempt to extract metadata about figures and tables.",
                    "default": False
                },
                "extract_formulas": {
                    "type": "boolean",
                    "description": "Attempt to extract formulas in MathML or LaTeX format.",
                    "default": True
                }
            },
            "required": ["arxiv_id"]
        }
    }
}

# ==============================================================================
# Implementations
# ==============================================================================

def search_arxiv_impl(params: SearchArxivParams) -> Dict[str, Any]:
    """Implementation of arXiv search using the official arXiv API."""
    import urllib.request
    import xml.etree.ElementTree as ET
    
    base_url = 'http://export.arxiv.org/api/query?'
    query_params = {
        'search_query': params.query,
        'start': 0,
        'max_results': params.max_results,
        'sortBy': params.sort_by,
        'sortOrder': params.sort_order
    }
    
    url = base_url + urllib.parse.urlencode(query_params)
    try:
        response = urllib.request.urlopen(url)
        data = response.read()
        root = ET.fromstring(data)
        
        results = []
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
            title = entry.find('atom:title', ns).text.replace('\\n', ' ').strip()
            summary = entry.find('atom:summary', ns).text.replace('\\n', ' ').strip()
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            published = entry.find('atom:published', ns).text
            
            results.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "published": published,
                "summary": summary
            })
            
        return {"success": True, "data": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


def scrape_arxiv_paper_impl(params: ScrapeArxivPaperParams) -> Dict[str, Any]:
    """
    Implementation of arXiv paper scraping. 
    In a real scenario, this uses BeautifulSoup4 (HTML) or PyMuPDF (PDF fallback).
    """
    # NOTE: This is a mocked implementation wrapper.
    # The actual scraping logic (HTML -> MathML / PDF fallback) as defined in PRD
    # would be integrated here.
    
    # Placeholder for actual scraping logic
    import time
    
    arxiv_id = params.arxiv_id
    html_url = f"https://arxiv.org/html/{arxiv_id}"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        # 1. Attempt to fetch HTML version
        # response = requests.get(html_url)
        # if response.status_code == 200:
        #     # Parse HTML with BeautifulSoup
        #     # Convert LaTeX to MathML
        #     return {"success": True, "format": "html", "content": extracted_text}
        
        # 2. Fallback to PDF if HTML is unavailable
        # response = requests.get(pdf_url)
        # if response.status_code == 200:
        #     # Parse PDF with PyMuPDF
        #     # Replace image-based formulas with '[FORMULA: image-only, skipped]'
        #     return {"success": True, "format": "pdf", "content": extracted_text}
        
        # Mock success response
        return {
            "success": True, 
            "arxiv_id": arxiv_id,
            "format": "html",
            "metadata": {
                "title": f"Sample Title for {arxiv_id}",
                "authors": ["Author A", "Author B"]
            },
            "content": "This is the full text of the paper extracted from HTML. [FORMULA: extracted MathML]. Section 1..."
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def search_semantic_scholar_impl(params: SearchSemanticScholarParams) -> Dict[str, Any]:
    """Implementation of Semantic Scholar Graph API search."""
    import requests
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    query_params = {
        "query": params.query,
        "limit": params.limit,
        "fields": "title,authors,year,abstract"
    }
    try:
        response = requests.get(url, params=query_params, timeout=10)
        if response.status_code == 200:
            return {"success": True, "data": response.json().get("data", [])}
        return {"success": False, "error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def search_duckduckgo_impl(params: SearchDuckDuckGoParams) -> Dict[str, Any]:
    """Lightweight implementation of DuckDuckGo Lite search via HTML parsing."""
    import requests
    from bs4 import BeautifulSoup
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    data = {"q": params.query}
    try:
        response = requests.post(url, headers=headers, data=data, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            
            # Find all result anchors
            anchors = soup.find_all("a", class_="result__url", limit=params.max_results)
            snippets = soup.find_all("a", class_="result__snippet", limit=params.max_results)
            
            for a, s in zip(anchors, snippets):
                results.append({
                    "url": a.get("href"),
                    "snippet": s.text.strip()
                })
            return {"success": True, "data": results}
        return {"success": False, "error": f"DDG Error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ==============================================================================
# Tool Execution Registry & Validation
# ==============================================================================

TOOLS_REGISTRY = {
    "search_arxiv": {
        "schema": SEARCH_ARXIV_SCHEMA,
        "implementation": search_arxiv_impl,
        "validator": SearchArxivParams
    },
    "scrape_arxiv_paper": {
        "schema": SCRAPE_ARXIV_PAPER_SCHEMA,
        "implementation": scrape_arxiv_paper_impl,
        "validator": ScrapeArxivPaperParams
    },
    "search_semantic_scholar": {
        "schema": SEARCH_SEMANTIC_SCHOLAR_SCHEMA,
        "implementation": search_semantic_scholar_impl,
        "validator": SearchSemanticScholarParams
    },
    "search_duckduckgo": {
        "schema": SEARCH_DUCKDUCKGO_SCHEMA,
        "implementation": search_duckduckgo_impl,
        "validator": SearchDuckDuckGoParams
    }
}

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Validates arguments and executes the specified tool safely."""
    if tool_name not in TOOLS_REGISTRY:
        return {"success": False, "error": {"code": "UNKNOWN_TOOL", "message": f"Tool '{tool_name}' not found."}}
        
    tool = TOOLS_REGISTRY[tool_name]
    
    try:
        # Validate arguments using Pydantic
        validated_args = tool["validator"](**arguments)
    except ValidationError as e:
        return {"success": False, "error": {"code": "VALIDATION_ERROR", "message": str(e)}}
        
    try:
        # Execute tool
        result = tool["implementation"](validated_args)
        return result
    except Exception as e:
        return {"success": False, "error": {"code": "EXECUTION_ERROR", "message": str(e)}}

# Example LLM Tool Calls (for reference)
# {
#     "name": "search_arxiv",
#     "parameters": {
#         "query": "ti:transformers",
#         "max_results": 3
#     }
# }
# 
# {
#     "name": "scrape_arxiv_paper",
#     "parameters": {
#         "arxiv_id": "1706.03762"
#     }
# }
