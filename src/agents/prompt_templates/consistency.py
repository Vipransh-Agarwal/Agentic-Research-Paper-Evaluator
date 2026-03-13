import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Output Contract (JSON Schema via Pydantic)
# ---------------------------------------------------------

class ConsistencyIssue(BaseModel):
    severity: Literal["critical", "major", "minor"] = Field(..., description="Severity of the issue")
    section: str = Field(..., description="Section of the paper where the issue was found")
    description: str = Field(..., description="Description of the logical or structural consistency issue")
    suggestion: str = Field(..., description="Actionable suggestion to resolve the issue")

class ConsistencyEvaluation(BaseModel):
    summary: str = Field(..., description="Brief 1-2 sentence overview of the paper's consistency")
    issues: List[ConsistencyIssue] = Field(default_factory=list, description="List of consistency issues")
    strengths: List[str] = Field(default_factory=list, description="List of positive consistency aspects")
    consistency_score: int = Field(..., ge=0, le=100, description="Overall consistency score from 0-100")

# ---------------------------------------------------------
# Prompt Variables Validation
# ---------------------------------------------------------

class ConsistencyPromptVariables(BaseModel):
    paper_title: str
    paper_abstract: str
    paper_text: str
    focus_area: Optional[str] = "Logical flow, mathematical consistency, and structural integrity"
    current_utc_time: str

# ---------------------------------------------------------
# Few-Shot Examples
# ---------------------------------------------------------

FEW_SHOT_EXAMPLES = """
## Example 1: Good Consistency

**Input:**
Title: "A Novel approach to X"
Abstract: "We propose X to solve Y. Our results show a 20% improvement."
Text: "Section 1: X is defined as... Section 2: Methodology for X involves... Section 3: Results show a 20% improvement..."

**Output:**
```json
{
  "summary": "The paper maintains strong logical flow and aligns its abstract claims perfectly with its methodology and results.",
  "issues": [],
  "strengths": [
    "Abstract claims match the results section perfectly.",
    "Methodology logically follows the problem statement."
  ],
  "consistency_score": 95
}
```

## Example 2: Inconsistent Paper

**Input:**
Title: "Scalable Systems"
Abstract: "We present a system that scales linearly to 1000 nodes."
Text: "Section 3: Due to bottleneck Z, the system cannot scale beyond 100 nodes."

**Output:**
```json
{
  "summary": "The paper contains a critical contradiction between its abstract claims and the reported results.",
  "issues": [
    {
      "severity": "critical",
      "section": "Abstract vs Section 3",
      "description": "Abstract claims linear scaling to 1000 nodes, but Section 3 explicitly states it cannot scale beyond 100 nodes due to bottleneck Z.",
      "suggestion": "Revise the abstract to accurately reflect the 100-node limitation, or provide methodology that actually achieves 1000 nodes."
    }
  ],
  "strengths": [
    "Clearly identifies the bottleneck Z."
  ],
  "consistency_score": 40
}
```
"""

# ---------------------------------------------------------
# Version Control
# ---------------------------------------------------------

CONSISTENCY_PROMPTS = {
    "v1.0": {
        "system": lambda vars: f"""
# System Prompt: Research Paper Consistency Evaluator

Current UTC Time: {vars.current_utc_time} (All your analysis must be context-aware of this date).

You are an expert academic peer-reviewer specializing in logical and structural consistency. Your role is to evaluate academic papers for logical flow, structural integrity, and consistency between claims (especially in the abstract/introduction) and the actual methodology or results.

## Output Format

Provide your evaluation strictly in the following JSON structure:

```json
{{
  "summary": "Brief 1-2 sentence overview",
  "issues": [
    {{
      "severity": "critical|major|minor",
      "section": "Section name or number",
      "description": "Description of the logical or structural consistency issue",
      "suggestion": "Actionable suggestion to resolve the issue"
    }}
  ],
  "strengths": ["List of positive consistency aspects"],
  "consistency_score": 0-100 (integer between 0 and 100)
}}
```

## Style Guidelines
- Be rigorous, objective, and specific.
- Cite specific sections for issues.
- Provide actionable suggestions.
- Balance criticism with praise.
- Use a professional, academic tone.

## Constraints
- Do NOT evaluate novelty or factuality (other agents handle this).
- Do focus on internal contradictions, broken logical chains, or unsupported claims within the text itself.
- Do NOT be overly pedantic about formatting or grammar.
- Ensure the output is valid JSON matching the schema.
""",
        "user": lambda vars: f"""
Please evaluate the consistency of the following research paper:

Title: {vars.paper_title}
Focus Area: {vars.focus_area}

Abstract:
{vars.paper_abstract}

Full Text / Chunk:
{vars.paper_text}

Provide a thorough consistency review following the output format specified in the system prompt.

{FEW_SHOT_EXAMPLES}
"""
    }
}

def build_consistency_prompt(vars_dict: dict, version: str = "v1.0") -> dict:
    """Builds the system and user prompts for Consistency Evaluation."""
    valid_vars = ConsistencyPromptVariables(**vars_dict)
    
    if version not in CONSISTENCY_PROMPTS:
        raise ValueError(f"Version {version} not found in CONSISTENCY_PROMPTS")
        
    template = CONSISTENCY_PROMPTS[version]
    return {
        "system": template["system"](valid_vars),
        "user": template["user"](valid_vars)
    }

def parse_consistency_output(output: str) -> ConsistencyEvaluation:
    """Parses and validates the LLM JSON output."""
    try:
        clean_output = output.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_output)
        return ConsistencyEvaluation(**parsed)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Output validation failed: {e}")
