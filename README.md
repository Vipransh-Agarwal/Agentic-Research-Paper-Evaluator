# Agentic Research Paper Evaluator

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Framework: LangGraph](https://img.shields.io/badge/Framework-LangGraph-FF6F00?logo=langchain&logoColor=white)](https://langchain.com/)
[![LLM: Gemini / OpenRouter](https://img.shields.io/badge/LLM-Gemini%20%7C%20OpenRouter-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-agent AI system that autonomously scrapes an arXiv link, decomposes the paper, and executes a comprehensive peer-review simulation across specialized domains to produce a detailed **Judgement Report**.

## 🚀 Key Features

- ✨ **Autonomous Scraping**: Extracts raw text natively from arXiv HTML endpoints, with a robust PDF fallback using PyMuPDF.
- 🤖 **Multi-Agent Evaluation**: Simulates peer-review across 5 specific domains:
    - **Consistency**: 0-100 score on logical flow.
    - **Grammar**: Categorical rating (High/Medium/Low).
    - **Novelty Index**: Qualitative impact description.
    - **Fact-Checking**: Interactive log of verified vs. unverified claims.
    - **Authenticity**: Percentage-based fabrication risk assessment.
- 🕒 **Context Aware**: All agents are UTC-time aware for accurate evaluation of "recent" vs "past" works.
- 🔒 **Context Window Safety**: Intelligently chunks documents to strictly enforce a maximum 16,000 token limit per LLM call.
- 🌡️ **Precision-Tuned**: Optimized temperatures (0.0 for facts, 0.2 for analysis) to ensure JSON reliability and factual rigor.
- 📊 **Rich Dashboard**: Streamlit-based UI with metric cards, color-coded Pass/Fail verdicts, and a visual Fact Check Log.

## 🛠️ Quick Start

```bash
git clone https://github.com/your-username/agentic-research-paper-evaluator.git
cd agentic-research-paper-evaluator
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY to .env
streamlit run app.py
```

## 📊 Dashboard Features

- 🖥️ **Real-time Status**: Track LangGraph node execution (Scrape -> Decompose -> Analyze -> Report) with live updates.
- 🚦 **Verdict Alert**: Immediate color-coded recommendation (Accept, Minor Revisions, Major Revisions, or Reject).
- ✅ **Fact Check Log**: Visual checklist of claims with icons (✅ supported, ❌ contradicted, ⚠️ needs verification).
- 📝 **Interactive Report**: Full Markdown rendering of the generated Judgement Report within the UI.

## 🏗️ Architecture

The system utilizes a modular, agent-centric architecture orchestrated by **LangGraph**:

```text
.
├── app.py                  # Streamlit Dashboard UI
├── requirements.txt        # Project dependencies
├── benchmarks/             # Reference judgement reports (e.g., Attention Is All You Need)
├── docs/                   # Detailed PRD and Architecture Design
├── reports/                # Generated Judgement Reports (.md & .pdf)
├── src/                    # Core source code
│   ├── main.py             # CLI Entrypoint
│   ├── agents/             # Agent definitions & prompt templates
│   │   └── prompt_templates/ 
│   │       ├── authenticity.py  # Fabrication risk synthesis
│   │       ├── consistency.py   # Logical flow (0-100)
│   │       ├── fact_checking.py # Claim verification (Temp 0.0)
│   │       ├── grammar.py       # categorical rating
│   │       └── novelty.py       # Qualitative description
│   ├── orchestrator/       # LangGraph workflow orchestration (UTC aware)
│   ├── output/             # Report aggregation & JSON extraction
│   ├── processing/         # 16k token chunking logic
│   └── scraper/            # arXiv HTML/PDF extraction engine
└── tests/                  # Pytest suite (11+ passed tests)
```

## 🧪 Development & Testing

### Running Tests
The project includes a comprehensive test suite covering scrapers, agents, and end-to-end workflows:
```bash
python -m pytest
```

### Running Benchmarks
Evaluate a foundational paper (e.g., Transformer paper) to verify system quality:
```bash
python scripts/run_benchmark.py
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Made with ❤️ by the Agentic AI Team
