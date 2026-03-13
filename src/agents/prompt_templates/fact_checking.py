import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Output Contract
# ---------------------------------------------------------

class FactCheckClaim(BaseModel):
    claim: str = Field(..., description="The factual claim extracted from the text")
    verdict: Literal["supported", "unsupported", "contradicted", "needs_verification"] = Field(..., description="The verdict on the claim's factual accuracy")
    evidence: str = Field(..., description="Evidence supporting the verdict, or reason why it needs verification")
    confidence: Literal["high", "medium", "low"] = Field(..., description="Confidence in this evaluation")

class FactCheckingEvaluation(BaseModel):
    summary: str = Field(..., description="Brief overview of the paper's factual accuracy")
    claims_evaluated: List[FactCheckClaim] = Field(default_factory=list, description="List of evaluated claims")
    fabrication_risk_score: int = Field(..., ge=1, le=10, description="Risk score for fabricated or hallucinated data/citations (1=low risk, 10=high risk)")
    fact_score: int = Field(..., ge=1, le=10, description="Overall factual correctness score from 1-10")

# ---------------------------------------------------------
# Prompt Variables Validation
# ---------------------------------------------------------

class FactCheckingPromptVariables(BaseModel):
    paper_text: str
    extract_count: Optional[int] = 5
    external_knowledge_allowed: Optional[bool] = True

# ---------------------------------------------------------
# Few-Shot Examples
# ---------------------------------------------------------

FEW_SHOT_EXAMPLES = """
## Example 1: High Accuracy

**Input:**
Text: "As established by Einstein's theory of general relativity (1915), massive objects cause a distortion in space-time."

**Output:**
```json
{
  "summary": "The text relies on well-established scientific facts and historical milestones.",
  "claims_evaluated": [
    {
      "claim": "Einstein's theory of general relativity was published in 1915 and states massive objects distort space-time.",
      "verdict": "supported",
      "evidence": "General relativity is a widely accepted scientific theory published in 1915.",
      "confidence": "high"
    }
  ],
  "fabrication_risk_score": 1,
  "fact_score": 10
}
```

## Example 2: Hallucinated / Fake Citation

**Input:**
Text: "Recent studies (Smith & Doe, 2024) have shown that drinking liquid mercury cures the common cold."

**Output:**
```json
{
  "summary": "The text contains dangerously false medical claims and potentially fabricated citations.",
  "claims_evaluated": [
    {
      "claim": "Drinking liquid mercury cures the common cold according to Smith & Doe, 2024.",
      "verdict": "contradicted",
      "evidence": "Liquid mercury is highly toxic and does not cure colds. The citation is likely fabricated to support a false claim.",
      "confidence": "high"
    }
  ],
  "fabrication_risk_score": 9,
  "fact_score": 2
}
```
"""

# ---------------------------------------------------------
# Version Control
# ---------------------------------------------------------

FACT_CHECKING_PROMPTS = {
    "v1.0": {
        "system": """
# System Prompt: Research Paper Fact-Checker

You are an expert fact-checker for academic research papers. Your role is to identify verifiable claims (historical dates, known scientific constants, established theories, citations) and evaluate their factual accuracy based on your broad internal knowledge base.

## Output Format

Provide your evaluation strictly in the following JSON structure:

```json
{
  "summary": "Brief overview of the paper's factual accuracy",
  "claims_evaluated": [
    {
      "claim": "The extracted claim",
      "verdict": "supported|unsupported|contradicted|needs_verification",
      "evidence": "Evidence or reasoning",
      "confidence": "high|medium|low"
    }
  ],
  "fabrication_risk_score": 1-10 (integer),
  "fact_score": 1-10 (integer)
}
```

## Style Guidelines
- Be highly skeptical and meticulous.
- Flag "needs_verification" if a claim is plausible but too niche for you to confidently verify.
- Give a high fabrication risk score for clearly hallucinated citations or physically impossible claims disguised as science.
- Use precise, analytical language.

## Constraints
- Focus exclusively on factual accuracy, NOT logical consistency (unless the math is factually wrong) or novelty.
- You must output valid JSON.
""",
        "user": lambda vars: f"""
Please fact-check the following text excerpt from a research paper. Identify up to {vars.extract_count} key verifiable claims and evaluate them.

External Knowledge Permitted: {vars.external_knowledge_allowed}

Text:
{vars.paper_text}

Provide a thorough fact-checking review following the output format specified in the system prompt.

{FEW_SHOT_EXAMPLES}
"""
    }
}

def build_fact_checking_prompt(vars_dict: dict, version: str = "v1.0") -> dict:
    valid_vars = FactCheckingPromptVariables(**vars_dict)
    if version not in FACT_CHECKING_PROMPTS:
        raise ValueError(f"Version {version} not found in FACT_CHECKING_PROMPTS")
    template = FACT_CHECKING_PROMPTS[version]
    return {
        "system": template["system"],
        "user": template["user"](valid_vars)
    }

def parse_fact_checking_output(output: str) -> FactCheckingEvaluation:
    try:
        clean_output = output.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_output)
        return FactCheckingEvaluation(**parsed)
    except Exception as e:
        raise ValueError(f"Output validation failed: {e}")
