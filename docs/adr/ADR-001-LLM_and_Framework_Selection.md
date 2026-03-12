# ADR-001: Selection of LLM Provider and Agentic Framework

**Status:** Accepted
**Date:** 2024-03-13
**Deciders:** Architecture Team
**Owner:** System Architect

## Context

The Agentic Research Paper Evaluator is designed to autonomously read, chunk, and critically evaluate academic papers from arXiv. This requires orchestrating multiple specialized AI agents (Consistency, Grammar, Novelty, Fact-Checking, Authenticity) to analyze chunks of text and synthesize a comprehensive "Judgement Report".

Two key architectural decisions needed to be made to support this workflow:
1. **LLM Provider:** We need a provider capable of handling complex reasoning over dense academic text. Crucially, the project constraints dictate the use of *free-tier* LLMs to keep operational costs at zero for individual researchers running the CLI locally.
2. **Agentic Framework:** We need a framework to manage the orchestration, state passing, prompt chaining, and error handling for the multi-agent workflow.

The constraints we are working under:
- Strict limit of 16,000 tokens per LLM call to prevent context dropping.
- Zero budget for LLM inference (MVP phase).
- Needs to be highly reliable despite the strict rate limits of free-tier APIs.

## Decision

We will use **Gemini API (Free Tier)** as the primary LLM provider and **LangGraph** (or a similarly state-graph-based framework like CrewAI, though LangGraph is preferred for fine-grained control over execution edges and pacing) as the agentic orchestrator.

## Alternatives Considered

### LLM Provider Selection

#### Option 1: Gemini API Free Tier (CHOSEN)
**Pros:**
- `gemini-2.0-flash` offers an massive 1M token context window (though we are artificially limiting to 16k chunks for reliability) and excellent reasoning capabilities.
- The free tier offers 1,500 requests per day and 1M Tokens Per Minute (TPM), which is highly generous for a free tier.
- Strong support for structured output (JSON).

**Cons:**
- Strict 15 Requests Per Minute (RPM) limit.

#### Option 2: OpenRouter (Free Models)
**Pros:**
- Access to a rotating cast of free open-source models (e.g., Llama 3, Mistral).
- No single API vendor lock-in.

**Cons:**
- Free models on OpenRouter are often heavily rate-limited, subject to queueing, or have smaller context windows (e.g., 8k) that might conflict with our 16k chunking strategy.
- Reasoning quality of free-tier models is often lower than Gemini Flash.

#### Option 3: Local LLMs via Ollama
**Pros:**
- Completely free and private. No rate limits whatsoever.

**Cons:**
- High hardware requirements for the end-user. Many researchers using the CLI on standard laptops will not be able to run an 8B+ parameter model fast enough to make the tool viable.
- Context window limitations on smaller local models.

### Agentic Framework Selection

#### Option 1: LangGraph (CHOSEN)
**Pros:**
- Explicit state management using a Graph structure.
- Allows for fine-grained control over execution paths (e.g., forcing a strictly sequential flow).
- **Crucial for this project:** Because Gemini has a 15 RPM limit, LangGraph allows us to easily inject an `asyncio.sleep(4)` between nodes in a sequential graph to naturally govern the pacing without complex queueing infrastructure.
- Deep integration with LangChain's ecosystem for tool use (Search APIs).

**Cons:**
- Steeper learning curve compared to higher-level abstractions.

#### Option 2: CrewAI
**Pros:**
- Excellent high-level abstractions for defining "Roles", "Goals", and "Backstories".
- Very fast to set up a standard multi-agent debate/workflow.

**Cons:**
- Less transparent execution flow ("black box" orchestration).
- Harder to enforce strict intra-agent time delays (like our required 4-second pacing) because CrewAI often tries to optimize for parallel execution behind the scenes.

#### Option 3: AutoGen
**Pros:**
- Great for conversational, multi-agent chat paradigms.

**Cons:**
- Better suited for code-generation and conversational tasks rather than the strict, document-chunk processing pipeline we require.

## Tradeoffs

**What we're optimizing for:**
- Cost ($0 per paper).
- Reliability of context extraction (strict 16k chunking).
- Stable orchestration that respects strict rate limits.

**What we're sacrificing:**
- Speed. By relying on Gemini Free's 15 RPM limit and enforcing a sequential graph with 4-second delays, a 5-agent evaluation of a 4-chunk paper will take ~3-5 minutes. We are trading execution speed for cost-efficiency.

## Consequences

### Positive
- The application can be distributed and run by anyone for free using a single API key.
- LangGraph provides a robust, visualizable state machine that makes debugging multi-chunk, multi-agent workflows much easier.
- The state graph architecture natively supports graceful degradation (e.g., if one agent fails, the graph can route to an error state or skip).

### Negative
- The 15 RPM limit is a severe bottleneck. The architecture is tightly coupled to this constraint (requiring artificial delays).
- If Google changes the Gemini Free tier limits, the core orchestration logic will need to be refactored or users will experience consistent `429 Too Many Requests` errors.

### Risks
- **Risk:** Users become frustrated by the 3-5 minute wait time per paper.
- **Mitigation:** The CLI must have excellent progress reporting. The LangGraph state updates will be streamed to the CLI so the user sees exactly what the system is currently analyzing.

## Implementation Notes
- Set `GEMINI_MODEL=gemini-2.0-flash` as the default in `.env.example`.
- Implement a sequential edge flow in LangGraph: `Scrape -> Decompose -> Consistency -> Grammar -> Novelty -> FactCheck -> Authenticity -> Report`.
- Wrap every LLM node in the graph with `tenacity` exponential backoff to handle the inevitable `429` errors if the pacing slightly drifts.
- Implement OpenRouter and Ollama as fallbacks inside the LangChain model initialization if Gemini fails exhaustively.

## Follow-up Actions
- [x] Create prototype LangGraph workflow (`src/orchestrator/workflow.py`).
- [ ] Implement robust logging in the CLI to show LangGraph state transitions.
- [ ] Write documentation for users on how to obtain their free Gemini API key.
