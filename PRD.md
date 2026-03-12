# Product Requirements Document

### 1. Overview

**Feature / Project Name:** Agentic Research Paper Evaluator

**Problem Statement:** The academic community and independent researchers are overwhelmed by the volume of papers published on platforms like arXiv. Manually verifying the internal consistency, factual accuracy, and novelty of a paper is time-consuming, and existing LLM tools often hallucinate or struggle with context limits when summarizing complex technical documents.

**Proposed Solution:** A multi-agent AI system that autonomously scrapes an arXiv link, decomposes the paper, and executes a comprehensive peer-review simulation across specialized domains to produce a detailed "Judgement Report".

**AI Build Summary:**
> Build a Python-based CLI or web application using an agentic framework (e.g., CrewAI, AutoGen, or LangGraph) that accepts an arXiv URL, scrapes the content, chunks it (max 16k tokens per LLM call), and uses specialized AI agents to evaluate consistency, grammar, novelty, facts, and authenticity. Output a structured Markdown or PDF report containing specific scores and logs. Use free-tier LLMs (Gemini Free, OpenRouter, or local models).

### 2. Goals & Success Metrics

**Primary Goal:** To provide researchers with an automated, reliable, and multi-faceted critical audit (Judgement Report) of any arXiv research paper to drastically reduce manual review time.

**Success Metrics:**
*   Successful extraction and parsing of text from >95% of provided valid arXiv URLs.
*   System successfully avoids LLM context-window limits by staying under the strict 16k token limit per call.
*   Generation of a complete, structurally sound Evaluation Report containing all 5 required metric domains for a test case.

**Anti-goals:**
*   Not trying to replace human peer-review entirely, but rather to simulate and assist it.
*   Not trying to build a new LLM from scratch; we are orchestrating existing free-tier LLM APIs.
*   Not trying to act as a general-purpose summarizer.

### 3. Scope & Constraints

**In scope:**
*   Web scraping of arXiv URLs to extract clean text/data.
*   Document decomposition (Abstract, Methodology, Results, Conclusion).
*   Multi-agent analysis covering: Consistency, Grammar, Novelty, Fact-Checking, and Authenticity.
*   Generation of a comprehensive Judgement Report (Markdown or PDF).
*   Integration with free-tier LLMs (Gemini, OpenRouter, local).
*   Optional enhancement: Mathpix OCR / LaTeX-OCR for extracting complex image-embedded formulas.

**Out of scope:**
*   Handling paywalled papers or non-arXiv repositories for the MVP.
*   Complex graphical UI (a CLI or basic web interface is sufficient).
*   Processing images, charts, or complex mathematical formatting beyond raw text extraction.

**Technical constraints:**
*   **Context Limit:** Strict limit of 16,000 tokens per LLM call.
*   **LLM Providers:** Must use free-tier providers or local models.
*   **Architecture:** Highly encouraged to use an agentic framework (CrewAI, AutoGen, LangGraph).

### 4. Jobs to Be Done (JTBD)

| Priority | Job Statement |
| --- | --- |
| 1 | When evaluating a new arXiv submission, I want an autonomous system to verify its consistency and factual accuracy, so I can quickly decide if the paper is worth a deep manual read. |
| 2 | When processing a long technical paper, I want the system to chunk the text efficiently under 16k tokens, so I can get reliable analysis without LLM hallucinations or context dropping. |
| 3 | When reviewing the final output, I want a structured Judgement Report with specific scores, so I can objectively compare the quality and risks of multiple papers. |

### 5. User Stories

| ID | Role | Action | Benefit | JTBD Ref |
| --- | --- | --- | --- | --- |
| US1 | Researcher | I want to input an arXiv URL | so that the system can automatically scrape and extract the paper's text. | J1 |
| US2 | System | I want to decompose the paper into sections | so that I can send targeted, <16k token chunks to the LLMs. | J2 |
| US3 | Researcher | I want specialized agents to evaluate consistency, novelty, and facts | so that I get a multi-dimensional peer-review simulation. | J1 |
| US4 | Researcher | I want to receive a final Evaluation Report with an executive summary and detailed scores | so that I can quickly assess the paper's validity and risk of fabrication. | J3 |

### 6. Proposed Experience

**Design Direction:** A simple, efficient developer-focused experience. The primary interaction is via a Command Line Interface (CLI) or a minimal web form. The emphasis is on the richness and structure of the output report rather than complex UI interactions.

**Key Screens / States:**
*   **Input State:** User provides an arXiv URL and selects the target LLM provider (via `.env` or CLI flags).
*   **Processing State:** A progress indicator detailing the current active agent (e.g., "Scraping...", "Agent: Fact-Checking...", "Compiling Report...").
*   **Output State:** The system generates and saves `Judgement_Report.md` or `.pdf` to the local directory and prints a success message.
*   **Error State:** Clear error messages if the URL is invalid, scraping fails, or LLM API limits are reached.

**Interaction Model:**
1.  Run application via terminal: `python main.py --url <arxiv_url>`
2.  Observe progress logs of the multi-agent workflow.
3.  Open the resulting evaluation report file.

### 7. Component Inventory

*(For a Web UI / CLI Hybrid Approach)*

| Component | Type | Description | Linked Stories |
| --- | --- | --- | --- |
| `CLI Entrypoint` | Action | Accepts arXiv URL and configurations. | US1 |
| `Scraper Service` | Action | Downloads and parses HTML/PDF from arXiv. | US1 |
| `Chunking Engine` | Action | Splits text logically into <16k token chunks. | US2 |
| `Agent Orchestrator` | Layout/Logic | Manages CrewAI/LangGraph agents and data passing. | US3 |
| `Report Generator` | Action | Formats the multi-agent JSON output into MD/PDF. | US4 |

### 8. Data Models

```ts
interface PaperData {
  url: string;
  title: string;
  authors: string[];
  abstract: string;
  sections: {
    heading: string;
    content: string; // Must be < 16k tokens
  }[];
}

interface AgentEvaluation {
  agentRole: "Consistency" | "Grammar" | "Novelty" | "FactChecker" | "Authenticity";
  findings: string;
  score?: number | string; // e.g., 0-100, "High", "35% risk"
}

interface FabricationMetrics {
  claimVerificationRatio: number; // 40% weight (Ratio of unverified to total claims)
  logicalDisconnectPenalty: number; // 30% weight (Based on methodology vs results disconnect)
  citationIntegrityIndex: number; // 20% weight (Verification of sampled citations)
  methodologicalVaguenessScore: number; // 10% weight (Lack of reproducible details)
}

interface JudgementReport {
  executiveSummary: string;
  passFailRecommendation: "Pass" | "Fail" | "Needs Manual Review";
  detailedScores: {
    consistencyScore: number; // 0-100
    grammarRating: "High" | "Medium" | "Low";
    noveltyIndex: string; // Qualitative
    factCheckLog: { claim: string; verified: boolean; notes: string }[];
    fabricationProbability: number; // Percentage, aggregated from FabricationMetrics
    fabricationMetricsBreakdown: FabricationMetrics;
  };
}
```

### 9. API / Integration Surface

| Method | Path/Tool | Description | Auth Required | Response Shape |
| --- | --- | --- | --- | --- |
| GET | `https://arxiv.org/html/...` | Scrape paper content (Primary) | No | Raw HTML |
| GET | `https://arxiv.org/pdf/...` | Scrape paper content (Fallback) | No | PDF bytes |
| POST | `https://generativelanguage.googleapis.com/...` | Gemini API for LLM inference | Yes (API Key) | `LLMResponse` |
| POST | `https://openrouter.ai/api/v1/chat/completions` | OpenRouter API for LLM inference | Yes (API Key) | `LLMResponse` |
| LOCAL | `http://localhost:11434/api/generate` | Ollama local inference | No | `LLMResponse` |

**External Integrations (Novelty & Fact-Checking Agents):**
*   **Semantic Scholar Graph API:** Used for comprehensive academic literature search, finding related papers, and cross-referencing abstracts. (Free REST API)
*   **arXiv Search API:** Used to pull metadata and abstracts of recent papers in specific domains to check for overlapping claims.
*   **Tavily / DuckDuckGo Search:** Supplementary web search to catch industry blog posts, GitHub repos, or non-arXiv preprints.

### 10. State Management Map

| State | Location | Persistence | Notes |
| --- | --- | --- | --- |
| `Paper Text` | Local Memory / Temp File | Session | Scraped content, chunked for processing. |
| `Agent Outputs` | Memory (Orchestrator State) | Session | Intermediate findings from each specialized agent. |
| `Final Report` | File System (`.md` / `.pdf`) | Persistent | The final compiled output for the user. |

### 11. Tech Stack Recommendation

**Backend / Core Logic:** Python 3.10+
**Agentic Framework:** LangGraph or CrewAI (CrewAI is excellent for strict role-based multi-agent setups).
**Scraping:** 
*   **Primary (HTML):** `BeautifulSoup4` for parsing `arxiv.org/html/<id>`.
*   **Fallback (PDF):** `PyMuPDF` or `pdfplumber` for parsing PDF endpoints when HTML returns a 404.
*   **Optional OCR:** Mathpix OCR or local `LaTeX-OCR` for complex image-embedded formulas.
**LLM Integration:** `langchain` or `litellm` (to easily switch between Gemini, OpenRouter, and Ollama), with `tenacity` for exponential backoff and caching mechanisms.
**Output Generation:** `markdown` and `reportlab` or `WeasyPrint` (for PDF generation).

### 12. Suggested File Structure

```text
.
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── main.py                 # CLI Entrypoint
│   ├── scraper/
│   │   └── arxiv_scraper.py    # URL fetching and text extraction
│   ├── processing/
│   │   └── chunker.py          # 16k token limit enforcement
│   ├── agents/
│   │   ├── consistency.py
│   │   ├── grammar.py
│   │   ├── novelty.py
│   │   ├── fact_checker.py
│   │   └── authenticity.py
│   ├── orchestrator/
│   │   └── workflow.py         # CrewAI/LangGraph graph definition
│   └── output/
│       └── report_builder.py   # MD/PDF compilation
└── tests/
    └── test_paper.py           # Evaluates a known arXiv paper
```

### 13. Acceptance Criteria

| Story ID | Criteria |
| --- | --- |
| US1 | [ ] CLI accepts an arXiv URL and validates its format. <br> [ ] System downloads and extracts raw text from the URL. |
| US2 | [ ] Extracted text is successfully parsed into distinct sections (Abstract, Methods, etc.). <br> [ ] A validation check proves no single chunk sent to the LLM exceeds 16,000 tokens. |
| US3 | [ ] 5 distinct agents execute their specific prompts on the text. <br> [ ] Agents return structured data matching their required scoring format. |
| US4 | [ ] System compiles all agent outputs into a unified report. <br> [ ] The report contains an Executive Summary, Consistency Score, Grammar Rating, Novelty Index, Fact Check Log, and Fabrication Probability. <br> [ ] Output is saved as a readable file (Markdown or PDF). |

### 14. Open Questions & Risks

**Resolved Architecture Decisions:**
1.  **Complex Mathematical Formulas:** Addressed via a 3-tier extraction pipeline. Primary HTML extraction uses ar5iv to convert LaTeX to MathML. Fallback PDF extraction logs `[FORMULA: image-only, skipped]` which agents treat as low-confidence. An optional Mathpix OCR / LaTeX-OCR enhancement provides ~98% image-formula coverage.
2.  **Gemini Free Tier RPM Limits:** The 15 RPM limit is managed without excessive delays by implementing a 4-second `asyncio.sleep` pacing, `tenacity` exponential backoff, a 3-tier provider fallback, caching chunk responses, and batching chunks <8k tokens.

**Open Questions:**
*   None currently identified.

**Risks:**
*   **Context Window Management:** Accurately counting tokens before sending to the LLM to strictly avoid the 16k limit.
*   **LLM Rate Limits:** Free-tier APIs (especially Gemini Free) have strict RPM limits (e.g., 15 RPM). The orchestrator must handle this by enforcing pacing (`asyncio.sleep(4)` between calls), implementing backoff/retries via `tenacity`, caching chunk responses, and batching small chunks.
*   **Hallucination in Fact-Checking:** The Fact-Checker agent might hallucinate "verified" facts if it relies purely on its internal weights rather than a search tool.
*   **API Rate Limits & Stability:** Relying heavily on free-tier APIs like Semantic Scholar Graph API and arXiv Search API may introduce bottlenecks if they have strict rate limits or experience downtime.

### 15. Bonus Sections

*   **Testing:** Provide a specific, known arXiv paper URL as a benchmark to ensure the pipeline runs end-to-end reliably before deploying.
*   **Extensibility:** Design the orchestrator so new agent roles (e.g., "Math Verifier" or "Citation Formatter") can be added effortlessly in the future.