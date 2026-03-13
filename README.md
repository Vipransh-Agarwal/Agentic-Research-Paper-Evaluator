# Agentic Research Paper Evaluator

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Framework: CrewAI/LangGraph](https://img.shields.io/badge/Framework-CrewAI%20%7C%20LangGraph-FF6F00?logo=langchain&logoColor=white)](https://langchain.com/)
[![LLM: Gemini / OpenRouter](https://img.shields.io/badge/LLM-Gemini%20%7C%20OpenRouter-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-agent AI system that autonomously scrapes an arXiv link, decomposes the paper, and executes a comprehensive peer-review simulation across specialized domains to produce a detailed "Judgement Report".

## Features

- ✨ **Autonomous Scraping**: Extracts raw text natively from arXiv HTML endpoints, with a robust PDF fallback using PyMuPDF.
- 🚀 **Multi-Agent Evaluation**: Simulates peer-review across 5 specific domains: Consistency, Grammar, Novelty, Fact-Checking, and Authenticity.
- 🔒 **Context Window Safety**: Intelligently chunks documents to strictly enforce a maximum 16,000 token limit per LLM call.
- 📦 **API Rate-Limit Resilience**: Built-in 4-second pacing, exponential backoff, and caching to operate reliably under Gemini's free-tier (15 RPM) limits.
- 📊 **Structured Judgement Report**: Generates a final `.md` or `.pdf` artifact containing an Executive Summary, detailed metrics, and a calculated Fabricability risk score.

## Quick Start

```bash
git clone https://github.com/your-username/agentic-research-paper-evaluator.git
cd agentic-research-paper-evaluator
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your Gemini API key to .env
python src/main.py --url https://arxiv.org/abs/2303.08774
```

## Installation

### Prerequisites

- Python 3.11 or higher
- Git

### From Source

```bash
# Clone the repository
git clone https://github.com/your-username/agentic-research-paper-evaluator.git
cd agentic-research-paper-evaluator

# Create and activate a virtual environment
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt
```

## Configuration

The system requires valid API keys to interact with external LLMs and search tools.

Create a `.env` file in the root directory (you can copy `.env.example`):

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GEMINI_API_KEY` | Your Google Gemini API key | Yes* | |
| `OPENROUTER_API_KEY` | Your OpenRouter API key | Yes* | |
| `TAVILY_API_KEY` | Your Tavily Search API key for Fact-Checking | Yes | |
| `SEMANTIC_SCHOLAR_API_KEY` | Your Semantic Scholar API key | No | |
| `LLM_MODEL` | The LLM model to use (LiteLLM format) | No | `gemini/gemini-3.1-flash-lite-preview` |
| `MAX_TOKENS_PER_CHUNK` | Maximum tokens per document chunk | No | `16000` |

*\* Choose at least one primary LLM provider (Gemini or OpenRouter).*

## Usage

### Command Line Interface

Run the application by providing a valid arXiv URL:

```bash
python src/main.py --url https://arxiv.org/abs/2303.08774
```

The system will display a progress indicator as it scrapes the paper, decomposes it, and orchestrates the multi-agent review process. Upon completion, a `Judgement_Report.md` (or PDF) will be saved in your current directory.

## Architecture

The system utilizes a modular, agent-centric architecture orchestrated by LangGraph/CrewAI:

```text
src/
├── main.py                 # CLI Entrypoint & Argument Parsing
├── scraper/                
│   └── arxiv_scraper.py    # URL fetching, HTML parsing (ar5iv), PDF fallback
├── processing/             
│   └── chunker.py          # 16k token limit enforcement & text sectioning
├── agents/                 
│   ├── prompt_templates/   # Agent-specific prompts
│   │   ├── consistency.py  # Checks logical flow between sections
│   │   ├── grammar.py      # Evaluates writing quality
│   │   ├── novelty.py      # Assesses uniqueness against prior work
│   │   ├── fact_checking.py# Verifies key claims
│   │   └── authenticity.py # Calculates Fabricability probability
├── orchestrator/           
│   └── workflow.py         # Agent graph definition and state management
└── output/                 
    └── extractor.py        # Report compilation into MD/PDF format
```

## Development

### Running Tests

To run the test suite against a known benchmark arXiv paper:

```bash
pytest tests/
```

### Extending Agents

To add a new evaluation dimension, create a new prompt template in `src/agents/prompt_templates/` and register the agent within `src/orchestrator/workflow.py`. Ensure the agent's output conforms to the structured data expectations of the `extractor.py`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️
