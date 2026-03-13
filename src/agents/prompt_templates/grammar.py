import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class GrammarIssue(BaseModel):
    severity: Literal["critical", "major", "minor"] = Field(..., description="Severity of the issue")
    description: str = Field(..., description="Description of the grammar or tone issue")
    suggestion: str = Field(..., description="Suggestion for improvement")

class GrammarEvaluation(BaseModel):
    summary: str = Field(..., description="Brief overview of the paper's grammar and professional tone")
    issues: List[GrammarIssue] = Field(default_factory=list, description="List of grammar issues found")
    grammar_rating: Literal["High", "Medium", "Low"] = Field(..., description="Overall grammar rating (High, Medium, or Low)")

class GrammarPromptVariables(BaseModel):
    paper_text: str
    current_utc_time: str

GRAMMAR_PROMPTS = {
    "v1.0": {
        "system": lambda vars: f"""
# System Prompt: Research Paper Grammar Evaluator

Current UTC Time: {vars.current_utc_time} (All your analysis must be context-aware of this date).

You are an expert academic copyeditor. Your role is to evaluate academic papers for grammatical correctness, spelling, and professional tone.

## Output Format

Provide your evaluation strictly in the following JSON structure:

```json
{{
  "summary": "Brief overview of grammar quality",
  "issues": [
    {{
      "severity": "critical|major|minor",
      "description": "Issue description",
      "suggestion": "How to fix it"
    }}
  ],
  "grammar_rating": "High|Medium|Low"
}}
```

## Constraints
- Focus only on grammar, spelling, punctuation, and academic tone.
- Do not evaluate scientific novelty or facts.
- Output valid JSON.
""",
        "user": lambda vars: f"""
Please evaluate the grammar and tone of the following text excerpt from a research paper:

Text:
{vars.paper_text}

Provide a thorough grammar review following the output format specified in the system prompt.
"""
    }
}

def build_grammar_prompt(vars_dict: dict, version: str = "v1.0") -> dict:
    valid_vars = GrammarPromptVariables(**vars_dict)
    if version not in GRAMMAR_PROMPTS:
        raise ValueError(f"Version {version} not found in GRAMMAR_PROMPTS")
    template = GRAMMAR_PROMPTS[version]
    return {
        "system": template["system"](valid_vars),
        "user": template["user"](valid_vars)
    }
