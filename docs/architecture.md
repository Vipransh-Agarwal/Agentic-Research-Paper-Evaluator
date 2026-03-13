# System Design: Agentic Research Paper Evaluator

## Overview

The system is a multi-agent AI pipeline designed to automate the peer-review process for arXiv research papers. It is built on a state-machine architecture using **LangGraph**, ensuring reliable data flow and specialized agent focus.

## High-Level Architecture

```text
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│    Scraper   │────▶│   Chunker    │
│  UI / CLI    │     │   (ar5iv)    │     │   (16k Lim)  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Final Report │     │  LangGraph   │     │ Specialized  │
│ (.md / .pdf) │◀────│ Orchestrator │◀───▶│   Agents     │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Core Components

### 1. Orchestrator (`src/orchestrator/workflow.py`)
*   **Engine:** LangGraph State Machine.
*   **State Management:** Maintains `GraphState` containing chunks, agent evaluations, final verdict, and overall score.
*   **Time Sync:** Injects `current_utc_time` into all agent prompts at runtime.
*   **Pacing:** Implements a 4-second delay between LLM calls to respect Gemini Free tier limits (15 RPM).

### 2. Specialized Agents (`src/agents/`)
*   **Consistency Agent (Temp 0.2):** Validates logical flow and section alignment. Outputs 0-100 score.
*   **Grammar Agent (Temp 0.2):** Evaluates professional tone and correctness. Outputs High/Medium/Low rating.
*   **Novelty Agent (Temp 0.2):** Assesses contribution via Semantic Scholar context. Outputs a qualitative Novelty Index.
*   **Fact-Checker (Temp 0.0):** Extracts and verifies claims using DuckDuckGo. Outputs a detailed Claim Log.
*   **Authenticity Synthesizer (Temp 0.2):** Aggregates all findings to calculate a final Fabrication Risk (%).

### 3. Extraction & Aggregation (`src/output/extractor.py`)
*   **Validation:** Uses Pydantic to strictly enforce JSON schemas from LLM responses.
*   **Weighted Averaging:** Calculates the `overall_score` based on consistency, accuracy, and grammar mappings.
*   **Verdict Logic:** Assigns a recommendation (Accept to Reject) based on the score/risk thresholds.

### 4. Scraper Service (`src/scraper/`)
*   **Primary:** Scrapes `arxiv.org/html/<id>` (via ar5iv) for clean LaTeX-to-MathML conversion.
*   **Fallback:** Scrapes PDF via `PyMuPDF` if HTML is missing.

## Data Flow

1.  **Input:** User provides an ArXiv URL.
2.  **Scrape:** System extracts raw text.
3.  **Chunk:** Text is split into sections, ensuring every chunk is < 16,000 tokens.
4.  **Evaluate:** LangGraph iterates through nodes (Consistency -> Grammar -> Novelty -> Fact-Check -> Authenticity).
5.  **Aggregate:** The `report` node compiles agent JSONs into a structured Pydantic object.
6.  **Export:** Generates Markdown/PDF and updates the Streamlit dashboard.

## Technical Decisions

*   **Temperature Splitting:** Fact-checking requires zero variance (`0.0`), while analytical synthesis benefits from slight flexibility (`0.2`).
*   **0-100 Scoring:** Shifted from 1-10 to provide higher precision for algorithmic risk assessments.
*   **UTC Injection:** Prevents agents from hallucinating about "recent" papers by giving them a grounded temporal reference.
*   **Schema Enforcement:** Every agent call is wrapped in a Pydantic validation layer to prevent pipeline breaks from malformed LLM outputs.
