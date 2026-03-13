import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Output Contract
# ---------------------------------------------------------

class NoveltyFinding(BaseModel):
    aspect: str = Field(..., description="The aspect of the paper (e.g., Methodology, Dataset, Application)")
    novelty_level: Literal["high", "medium", "low", "none"] = Field(..., description="Level of novelty for this aspect")
    justification: str = Field(..., description="Reasoning behind the novelty level, comparing to known baselines")

class NoveltyEvaluation(BaseModel):
    summary: str = Field(..., description="Brief overview of the paper's overall novelty")
    findings: List[NoveltyFinding] = Field(default_factory=list, description="Specific novelty findings")
    similar_works_referenced: bool = Field(..., description="Did the paper adequately reference similar works?")
    novelty_index: str = Field(..., description="A qualitative description summarizing the novelty and impact")

# ---------------------------------------------------------
# Prompt Variables Validation
# ---------------------------------------------------------

class NoveltyPromptVariables(BaseModel):
    paper_title: str
    paper_abstract: str
    paper_text: str
    domain_knowledge_context: Optional[str] = "General academic baseline"
    current_utc_time: str

# ---------------------------------------------------------
# Few-Shot Examples
# ---------------------------------------------------------

FEW_SHOT_EXAMPLES = """
## Example 1: High Novelty

**Input:**
Title: "Quantum Gravity via Neural Networks"
Abstract: "We introduce a completely new framework mapping neural network weights to quantum states, solving a 30-year-old open problem."
Domain Context: General Physics

**Output:**
```json
{
  "summary": "The paper introduces a highly novel, paradigm-shifting framework bridging machine learning and quantum physics.",
  "findings": [
    {
      "aspect": "Methodology",
      "novelty_level": "high",
      "justification": "Mapping NN weights to quantum states to solve this specific open problem has not been proposed in existing literature."
    }
  ],
  "similar_works_referenced": true,
  "novelty_index": "Breakthrough: Establishes a new cross-disciplinary methodology with high potential impact."
}
```

## Example 2: Low Novelty

**Input:**
Title: "Image Classification using ResNet50"
Abstract: "We apply the standard ResNet50 architecture to classify images of common house pets."
Domain Context: Computer Vision

**Output:**
```json
{
  "summary": "The paper applies a well-known, existing architecture to a standard problem without introducing new methodologies.",
  "findings": [
    {
      "aspect": "Methodology",
      "novelty_level": "none",
      "justification": "ResNet50 is a standard, off-the-shelf architecture."
    },
    {
      "aspect": "Application",
      "novelty_level": "low",
      "justification": "Classifying house pets is a common benchmark task in computer vision."
    }
  ],
  "similar_works_referenced": true,
  "novelty_index": "Incremental: Primarily an application of existing methods to a well-studied problem."
}
```
"""

# ---------------------------------------------------------
# Version Control
# ---------------------------------------------------------

NOVELTY_PROMPTS = {
    "v1.0": {
        "system": lambda vars: f"""
# System Prompt: Research Paper Novelty Evaluator

Current UTC Time: {vars.current_utc_time} (All your analysis must be context-aware of this date).

You are an expert academic peer-reviewer specializing in assessing the novelty and originality of research papers. Your role is to determine if the paper introduces new ideas, methods, datasets, or applications, or if it merely iterates on existing work.

## Output Format

Provide your evaluation strictly in the following JSON structure:

```json
{{
  "summary": "Brief 1-2 sentence overview of the paper's novelty",
  "findings": [
    {{
      "aspect": "Aspect of the paper (e.g., Methodology, Dataset)",
      "novelty_level": "high|medium|low|none",
      "justification": "Reasoning comparing to known baselines"
    }}
  ],
  "similar_works_referenced": true|false,
  "novelty_index": "Qualitative description summarizing novelty"
}}
```

## Style Guidelines
- Be objective and context-aware.
- Justify novelty claims based on typical domain knowledge.
- Avoid penalizing papers just for being applied research, but accurately reflect the level of methodological novelty.
- Use professional, academic tone.

## Constraints
- Do NOT evaluate grammatical correctness or internal logical consistency.
- Focus strictly on originality and contribution to the field.
- Ensure valid JSON output.
""",
        "user": lambda vars: f"""
Please evaluate the novelty of the following research paper:

Title: {vars.paper_title}
Domain Context: {vars.domain_knowledge_context}

Abstract:
{vars.paper_abstract}

Full Text / Chunk:
{vars.paper_text}

Provide a thorough novelty review following the output format specified in the system prompt.

{FEW_SHOT_EXAMPLES}
"""
    }
}

def build_novelty_prompt(vars_dict: dict, version: str = "v1.0") -> dict:
    valid_vars = NoveltyPromptVariables(**vars_dict)
    if version not in NOVELTY_PROMPTS:
        raise ValueError(f"Version {version} not found in NOVELTY_PROMPTS")
    template = NOVELTY_PROMPTS[version]
    return {
        "system": template["system"](valid_vars),
        "user": template["user"](valid_vars)
    }

def parse_novelty_output(output: str) -> NoveltyEvaluation:
    try:
        clean_output = output.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_output)
        return NoveltyEvaluation(**parsed)
    except Exception as e:
        raise ValueError(f"Output validation failed: {e}")
