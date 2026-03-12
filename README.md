# Agentic Research Paper Evaluator

A multi-agent AI system that autonomously scrapes an arXiv link, decomposes the paper, and executes a comprehensive peer-review simulation across specialized domains to produce a detailed "Judgement Report".

## Features
- Scrapes arXiv HTML natively (with PDF fallback).
- Evaluates Papers for Consistency, Grammar, Novelty, and Accuracy.
- Generates a Fabricability risk score.
- Operates under strict 16k token limit window chunks.

## Tech Stack
- Python 3.10+
- LangGraph / CrewAI
- BeautifulSoup4
- Gemini / OpenRouter API

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your_repo_url>
   cd agentic-research-paper-evaluator
   ```

2. **Set up Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and fill in your API keys.
   ```bash
   cp .env.example .env
   ```

## Usage

Run the main CLI script with an arXiv URL:
```bash
python src/main.py --url https://arxiv.org/abs/2303.08774
```
