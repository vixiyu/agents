import os
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

MODEL = "gpt-4.1-mini"

TONE_GUIDES = {
    "Casual & friendly": "Friendly, natural, not overly formal. Light warmth.",
    "Polished & professional": "Professional, clear, confident. No slang.",
    "Executive concise": "Very concise, high-signal, minimal adjectives. Direct.",
    "Firm but polite": "Clear boundaries, respectful, no apology unless needed.",
    "Follow-up / nudge": "Brief follow-up. Assume positive intent. Clear next step.",
    "Apology / repair": "Own the issue, concise apology, propose fix, move forward.",
    "Ask / request": "Clear ask, context in 1–2 lines, polite close, easy to say yes/no.",
    "Negotiation / boundary": "Calm, confident, propose terms, invite alternatives.",
}

SYSTEM_PROMPT = """You are an expert communications assistant.

Rules:
- Preserve facts; do not invent details.
- Keep names, dates, dollar amounts, and commitments unchanged.
- If something is missing, keep placeholders like [DATE], [AMOUNT], [NAME] rather than guessing.
- Avoid sounding needy, defensive, or overly apologetic unless tone requires it.
- Output MUST be valid JSON only. No extra text.

Return JSON with keys:
- subjects: array of 3 subject lines (or [] if not requested)
- versions: array of 3 objects: {label, body}
- notes: array of bullet strings
"""

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = FastAPI(title="Email Tone Formatter API")

class Flags(BaseModel):
    shorten: bool = False
    subject_lines: bool = True
    more_warm: bool = False
    more_firm: bool = False
    less_hedgy: bool = True

class RewriteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    tone: str
    goal: Optional[str] = None
    flags: Flags = Flags()

class Version(BaseModel):
    label: str
    body: str

class RewriteResponse(BaseModel):
    subjects: List[str]
    versions: List[Version]
    notes: List[str]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest):
    if req.tone not in TONE_GUIDES:
        raise HTTPException(status_code=400, detail=f"Unknown tone. Choose one of: {list(TONE_GUIDES.keys())}")

    constraints = []
    if req.flags.shorten:
        constraints.append("Shorten by ~30% while preserving meaning.")
    constraints.append(f"Tone guide: {TONE_GUIDES[req.tone]}")

    if req.flags.more_warm:
        constraints.append("Slightly warmer and more personable (without becoming verbose).")
    if req.flags.more_firm:
        constraints.append("More direct and confident; reduce softening language.")
    if req.flags.less_hedgy:
        constraints.append("Remove hedging and apologetic phrasing (e.g., 'just', 'sorry to bother', 'I was wondering').")

    payload: Dict[str, Any] = {
        "draft": req.text,
        "tone": req.tone,
        "goal": req.goal,
        "constraints": constraints,
        "include_subjects": req.flags.subject_lines,
        "output_schema": {
            "subjects": "3 subject lines (or empty list if include_subjects is false)",
            "versions": [
                {"label": "Option A", "body": "string"},
                {"label": "Option B", "body": "string"},
                {"label": "Option C", "body": "string"},
            ],
            "notes": ["string", "string"],
        },
    }

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0.4,
    )

    try:
        out = json.loads(resp.choices[0].message.content)
    except Exception:
        raise HTTPException(status_code=500, detail="Model returned non-JSON. Try again.")

    # Normalize subjects when subject_lines is false
    subjects = out.get("subjects", [])
    if not req.flags.subject_lines:
        subjects = []

    versions_raw = out.get("versions", [])
    versions = [{"label": v.get("label", "Option"), "body": v.get("body", "")} for v in versions_raw][:3]
    notes = out.get("notes", [])

    return {"subjects": subjects, "versions": versions, "notes": notes}
