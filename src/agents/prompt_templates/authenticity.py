import json
from typing import List, Optional
from pydantic import BaseModel, Field

class FabricationMetrics(BaseModel):
    claimVerificationRatio: float = Field(..., description="Estimated percentage of verified claims vs unverified (0-100)")
    logicalDisconnectPenalty: float = Field(..., description="Penalty score for logical inconsistencies (0-100)")
    citationIntegrityIndex: float = Field(..., description="Index representing citation reliability based on fact check (0-100)")
    methodologicalVaguenessScore: float = Field(..., description="Score for vague methodologies (0-100)")

class AuthenticityEvaluation(BaseModel):
    summary: str = Field(..., description="Brief overview of the paper's authenticity and fabrication risk")
    fabrication_probability: float = Field(..., description="Calculated final risk of fabrication (percentage 0-100%)")
    metrics: FabricationMetrics = Field(..., description="Detailed fabrication metrics")

class AuthenticityPromptVariables(BaseModel):
    consistency_summary: str
    grammar_summary: str
    novelty_summary: str
    fact_check_summary: str
    current_utc_time: str

AUTHENTICITY_PROMPTS = {
    "v1.0": {
        "system": lambda vars: f"""
# System Prompt: Research Paper Authenticity Synthesizer

Current UTC Time: {vars.current_utc_time} (All your analysis must be context-aware of this date).

You are a master synthesis agent evaluating the overall authenticity and risk of fabrication of an academic paper. 
You will receive summaries from specialized review agents (Consistency, Grammar, Novelty, Fact-Checking). 
Your task is to synthesize these findings and calculate a final fabrication probability score and detailed metrics.

## Output Format

Provide your evaluation strictly in the following JSON structure:

```json
{{
  "summary": "Synthesis of the paper's authenticity based on the 4 agent reports.",
  "fabrication_probability": 0.0-100.0,
  "metrics": {{
    "claimVerificationRatio": 0.0-100.0,
    "logicalDisconnectPenalty": 0.0-100.0,
    "citationIntegrityIndex": 0.0-100.0,
    "methodologicalVaguenessScore": 0.0-100.0
  }}
}}
```

## Guidelines
- If fact-checking reports hallucinated citations or contradicted claims, fabrication probability and logical disconnect should be high.
- If consistency is low, methodological vagueness should be higher.
- Ensure the output is valid JSON.
""",
        "user": lambda vars: f"""
Based on the following evaluations from specialized agents, calculate the final authenticity metrics.

1. Consistency Agent Findings:
{vars.consistency_summary}

2. Grammar Agent Findings:
{vars.grammar_summary}

3. Novelty Agent Findings:
{vars.novelty_summary}

4. Fact-Checking Agent Findings:
{vars.fact_check_summary}

Provide a comprehensive authenticity synthesis following the JSON output format.
"""
    }
}

def build_authenticity_prompt(vars_dict: dict, version: str = "v1.0") -> dict:
    valid_vars = AuthenticityPromptVariables(**vars_dict)
    if version not in AUTHENTICITY_PROMPTS:
        raise ValueError(f"Version {version} not found in AUTHENTICITY_PROMPTS")
    template = AUTHENTICITY_PROMPTS[version]
    return {
        "system": template["system"](valid_vars),
        "user": template["user"](valid_vars)
    }
