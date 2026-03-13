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
- 📊 **Structured Judgement Report**: Generates a final `.md` artifact containing an Executive Summary, detailed metrics, and a calculated Fabricability risk score.

## Quick Start

```bash
git clone https://github.com/your-username/agentic-research-paper-evaluator.git
cd agentic-research-paper-evaluator
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your Gemini/OpenRouter API key to .env
streamlit run app.py
```

## Installation

### Prerequisites

- Python 3.11 or higher
- Git

### Setup for First-Time Users

1. **Clone & Navigate**:
   ```bash
   git clone https://github.com/your-username/agentic-research-paper-evaluator.git
   cd agentic-research-paper-evaluator
   ```

2. **Environment Setup**:
   Create a virtual environment to keep dependencies isolated.
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **API Configuration**:
   Copy the example environment file and add your keys.
   ```bash
   cp .env.example .env
   ```
   Open `.env` and fill in `GEMINI_API_KEY` (or `OPENROUTER_API_KEY`).

5. **Launch the UI**:
   ```bash
   streamlit run app.py
   ```

## Usage

### Web Interface (Recommended)

The easiest way to use the evaluator is through the Streamlit dashboard:

1. Run `streamlit run app.py`.
2. Open the URL provided in your terminal (usually `http://localhost:8501`).
3. Enter a valid arXiv URL (e.g., `https://arxiv.org/abs/1706.03762`).
4. Click **Evaluate Paper** and watch the multi-agent orchestration in real-time.
5. View scores and the full Judgement Report directly in the browser.

### Command Line Interface

You can also run the evaluation directly from the terminal:

```bash
python src/main.py --url https://arxiv.org/abs/1706.03762
```

The system will display progress as it scrapes and reviews the paper. A Markdown report will be generated in the `reports/` directory.

## Dashboard Features

- 🖥️ **Real-time Status**: Track LangGraph node execution (Scraping -> Analyzing -> Reporting) with live updates.
- 📊 **Metric Cards**: High-level scores for Consistency, Grammar, Novelty, and Factuality displayed in a clean dashboard.
- 📝 **Interactive Report**: Full Markdown rendering of the generated Judgement Report within the UI.
- 🛡️ **Risk Highlighting**: Clear visibility of the calculated Fabrication Risk score.

## Benchmarks & Existing Reports

The system has been benchmarked against foundational research papers. You can find pre-generated reports in the following locations:

- **Benchmark Papers**:
  - [Judgement_Report_1706.03762_Attention_Is_All_You_Need.md](./benchmarks/Judgement_Report_1706.03762_Attention_Is_All_You_Need.md)
- **Recently Generated Reports**:
  - [Judgement_Report_1706.03762.md](./reports/Judgement_Report_1706.03762.md)
  - [Judgement_Report_2603.11152.md](./reports/Judgement_Report_2603.11152.md)

Every evaluation run saves its final output to the `reports/` directory as a Markdown file for offline viewing.

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
    └── extractor.py        # Report compilation into MD
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
