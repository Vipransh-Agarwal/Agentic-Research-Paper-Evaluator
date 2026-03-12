# System Design: Agentic Research Paper Evaluator

## Overview

The Agentic Research Paper Evaluator is a multi-agent AI system designed to autonomously evaluate academic papers from arXiv. It scrapes the paper, decomposes it into logical chunks, and simulates a comprehensive peer-review process across specialized domains (Consistency, Grammar, Novelty, Fact-Checking, and Authenticity) to produce a detailed "Judgement Report". This tool aims to assist researchers by automating the initial critical audit of papers, reducing manual review time while strictly avoiding LLM context-window limits.

## Requirements

### Functional

- **Input:** Accept an arXiv URL via Command Line Interface (CLI).
- **Scraping:** Extract text from the arXiv URL (preferring HTML, with a PDF fallback).
- **Decomposition:** Parse the text logically into sections (e.g., Abstract, Methodology, Results, Conclusion).
- **Chunking:** Ensure no text chunk sent to an LLM exceeds the strict 16,000 token limit.
- **Multi-Agent Evaluation:** Execute 5 specialized AI agents:
  - Consistency Agent
  - Grammar & Language Agent
  - Novelty Agent (integrating Semantic Scholar / arXiv Search)
  - Fact-Checker Agent (integrating Tavily / DuckDuckGo)
  - Authenticity Agent (calculating Fabrication Probability)
- **Output:** Generate a comprehensive Judgement Report in Markdown or PDF format.

### Non-Functional

- **Context Limits:** Strictly adhere to the <16k token limit per LLM call to prevent context dropping and hallucinations.
- **Cost:** Utilize free-tier LLM providers (Gemini Free, OpenRouter) or local models (Ollama).
- **Reliability:** Successfully extract and parse text from >95% of valid arXiv URLs.
- **Resilience:** Handle LLM API rate limits with proper backoff and retry mechanisms.

## High-Level Architecture

```text
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  CLI Client  │────▶│    Scraper   │────▶│   Chunker    │
│  (main.py)   │     │   Service    │     │   Engine     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Final Report │     │    Agent     │     │ Specialized  │
│  (.md/.pdf)  │◀────│ Orchestrator │◀───▶│   Agents     │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Components

### 1. CLI Entrypoint (`src/main.py`)
**Responsibilities:**
- Accept arXiv URL and configuration flags from the user.
- Trigger the orchestrator workflow.
- Display progress to the user.

**Technology:** Python `argparse` or `typer`.

### 2. Scraper Service (`src/scraper/arxiv_scraper.py`)
**Responsibilities:**
- Implement a 3-tier extraction pipeline:
  - **Tier 1 (Primary):** Scrape `arxiv.org/html/<id>` via BeautifulSoup4 (ar5iv converts LaTeX math to MathML).
  - **Tier 2 (Fallback):** Fall back to `arxiv.org/pdf/<id>` using PyMuPDF if HTML is 404 or yields <500 words. Extracts inline text math and logs `[FORMULA: image-only, skipped]` for image-embedded formulas.
  - **Tier 3 (Optional Enhancement):** Mathpix OCR or local LaTeX-OCR for ~98% image-formula coverage.

**Technology:** `BeautifulSoup4` (HTML parsing), `PyMuPDF` or `pdfplumber` (PDF parsing), `Mathpix OCR` / `LaTeX-OCR` (Optional).

### 3. Chunking Engine (`src/processing/chunker.py`)
**Responsibilities:**
- Decompose raw text into logical sections.
- Count tokens to ensure chunks are under the 16,000 token limit.
- Split large sections if necessary while preserving semantic boundaries.

**Technology:** Python, tokenizer libraries (e.g., `tiktoken` or model-specific tokenizers).

### 4. Agent Orchestrator (`src/orchestrator/workflow.py`)
**Responsibilities:**
- Define agent roles, goals, and tasks.
- Manage the sequential or hierarchical workflow.
- Pass chunked data to the appropriate agents (batching small chunks <8k tokens to reduce request count).
- Enforce pacing (`asyncio.sleep(4)` between calls) to respect 15 RPM limits.
- Handle API rate limits with `tenacity` exponential backoff (max 3 retries) and a 3-tier provider fallback.
- Cache LLM responses per chunk hash to skip re-processing on pipeline retries.

**Technology:** `CrewAI` or `LangGraph`, `tenacity` (retries), `asyncio` (pacing).

### 5. Specialized Agents (`src/agents/`)
**Responsibilities:**
- **Consistency:** Validate methodology vs. results.
- **Grammar:** Assess professional tone and syntax.
- **Novelty:** Check for overlapping claims using external search tools.
- **Fact-Checker:** Extract and verify claims using web search.
- **Authenticity:** Calculate the Fabrication Probability based on predefined metrics.

**Technology:** `langchain` / `litellm` for LLM integration; specific agents will use tools interacting with Semantic Scholar Graph API, arXiv Search API, and Tavily/DuckDuckGo.

### 6. Report Generator (`src/output/report_builder.py`)
**Responsibilities:**
- Compile the JSON/structured outputs from all agents.
- Calculate final scores (e.g., Fabrication Probability).
- Format and export the final `Judgement_Report.md` or `.pdf`.

**Technology:** `markdown`, `reportlab` or `WeasyPrint`.

## Data Flow

1. **Input:** User runs `python main.py --url <arxiv_url>`.
2. **Scraping:** Scraper Service fetches the paper content (HTML or PDF) and extracts raw text.
3. **Processing:** Chunking Engine decomposes the text into logical sections and chunks it, strictly enforcing the 16k token limit.
4. **Orchestration:** Agent Orchestrator receives the chunks and initializes the CrewAI/LangGraph workflow.
5. **Evaluation:** Specialized Agents process the chunks:
   - Perform LLM inference for analysis.
   - Novelty and Fact-Checker agents query external APIs (Semantic Scholar, Tavily) for verification.
6. **Aggregation:** Agents return structured evaluations (JSON) back to the Orchestrator.
7. **Reporting:** Report Generator compiles the structured data, calculates final metrics, and saves the output to the local filesystem as a Markdown or PDF file.

## Data Model

```typescript
interface PaperData {
  url: string;
  title: string;
  authors: string[];
  abstract: string;
  sections: {
    heading: string;
    content: string; // Enforced < 16,000 tokens
  }[];
}

interface AgentEvaluation {
  agentRole: "Consistency" | "Grammar" | "Novelty" | "FactChecker" | "Authenticity";
  findings: string;
  score?: number | string;
}

interface FabricationMetrics {
  claimVerificationRatio: number;       // 40% weight
  logicalDisconnectPenalty: number;     // 30% weight
  citationIntegrityIndex: number;       // 20% weight
  methodologicalVaguenessScore: number; // 10% weight
}

interface JudgementReport {
  executiveSummary: string;
  passFailRecommendation: "Pass" | "Fail" | "Needs Manual Review";
  detailedScores: {
    consistencyScore: number;
    grammarRating: "High" | "Medium" | "Low";
    noveltyIndex: string;
    factCheckLog: { claim: string; verified: boolean; notes: string }[];
    fabricationProbability: number;
    fabricationMetricsBreakdown: FabricationMetrics;
  };
}
```

## API / Integration Surface

| Integration | Description | Auth Required |
| :--- | :--- | :--- |
| **arXiv HTML/PDF API** | Primary source for scraping paper content. | No |
| **Gemini API** | Primary LLM provider for agent reasoning. | Yes (API Key) |
| **OpenRouter API** | Alternative/Fallback LLM provider. | Yes (API Key) |
| **Ollama (Local)** | Local LLM inference. | No |
| **Semantic Scholar Graph API** | Used by Novelty Agent for literature cross-referencing. | No (Free tier) |
| **arXiv Search API** | Used by Novelty Agent to check overlapping claims. | No |
| **Tavily / DuckDuckGo Search** | Used by Fact-Checker Agent to verify claims via web. | Yes/No |

## Scaling Considerations

### Current Capacity (Local CLI)
- Designed for single-user execution on a local machine.
- Limited primarily by the rate limits of the free-tier LLM and Search APIs.

### 10x Scale (Basic Web Service)
- **Architecture Shift:** Move from synchronous CLI to an asynchronous backend (e.g., FastAPI + Celery + Redis).
- **Processing:** Since agentic evaluation takes time, users submit a URL and receive a Job ID to poll for the result.
- **Caching:** Implement a database (PostgreSQL) to cache previously generated reports for specific arXiv URLs to save API costs and time.
- **Rate Limiting:** Implement per-user rate limiting to protect LLM API key quotas.

### 100x Scale (SaaS Platform)
- **Distributed Workers:** Scale background workers horizontally based on queue depth.
- **LLM Load Balancing:** Implement round-robin or dynamic routing across multiple LLM provider accounts/keys or self-hosted models to bypass strict free-tier rate limits.
- **Vector Database:** Index scraped papers into a vector database (e.g., Pinecone, Qdrant) to drastically speed up internal novelty checks across the platform's history.

## Failure Modes

### 1. Scraping Failure
- **Impact:** Cannot process the paper.
- **Mitigation:** Robust HTML parsing with a reliable fallback to PDF extraction. Clear error messages if both fail (e.g., non-standard arXiv format).

### 2. LLM API Rate Limits (HTTP 429)
- **Impact:** Agent workflow halts mid-process.
- **Mitigation:** Implement exponential backoff and retry logic in the Orchestrator. Support fallback to alternative free-tier providers configured in `.env`.

### 3. Context Window Exceeded
- **Impact:** LLM returns an error or truncates output, leading to incomplete analysis or hallucinations.
- **Mitigation:** Strict, preventative token counting in the Chunking Engine before dispatching tasks to agents.

### 4. External Search APIs Unavailable
- **Impact:** Fact-Checking and Novelty agents fail to verify claims.
- **Mitigation:** Graceful degradation. The agent should note in its output that external verification was unavailable and rely purely on internal LLM knowledge, explicitly stating this limitation in the final report.

## Cost Estimate

Designed to operate entirely on free tiers for the MVP:
- **Compute:** $0 (Local execution)
- **LLM APIs:** $0 (Gemini Free tier, OpenRouter free models)
- **Search APIs:** $0 (Tavily free tier, DuckDuckGo, Semantic Scholar)
- **Total:** $0/month for individual CLI usage.

## Security

- API Keys (Gemini, OpenRouter, Tavily) are stored locally in a `.env` file and excluded from version control via `.gitignore`.
- The application processes public academic data and does not collect or store sensitive PII.

## Resolved Architecture Decisions

### 1. Complex Mathematical Formulas
**Resolution:** Implemented a 3-tier extraction pipeline. Primary HTML extraction converts LaTeX to MathML. Fallback PDF extraction logs `[FORMULA: image-only, skipped]` for image-embedded formulas, which agents are instructed to treat as low-confidence. An optional Mathpix OCR / LaTeX-OCR enhancement provides ~98% image-formula coverage.

### 2. Gemini Free Tier RPM Limits
**Resolution:** The `gemini-2.0-flash` model (15 RPM) is viable without extensive artificial delays. Mitigation strategy:
- 4-second `asyncio.sleep` between agent calls.
- `tenacity` exponential backoff (max 3 retries) on HTTP 429.
- 3-tier provider fallback (Gemini Flash → OpenRouter free tier → Ollama).
- Caching LLM responses per chunk hash.
- Batching small chunks (<8k tokens) to reduce request count.
