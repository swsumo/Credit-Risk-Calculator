import os
import re
from typing import Optional

from groq import Groq

_client: Optional[Groq] = None

SYSTEM = """You are a senior credit risk analyst at a lending institution.
Explain model predictions in plain, professional English — no jargon, no formulas.
Be concise (3-5 sentences or a short numbered list). Give direct, actionable insights."""


def _client_get() -> Groq:
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY not set")
        _client = Groq(api_key=key)
    return _client


def _clean(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **bold** → plain
    text = re.sub(r'\*(.+?)\*',     r'\1', text)    # *italic* → plain
    text = re.sub(r'#{1,6}\s+',     '',    text)    # ## headings → plain
    text = re.sub(r'\n{3,}', '\n\n', text)          # collapse triple newlines
    return text.strip()


def _chat(user: str, history: list[dict] | None = None, temp: float = 0.4) -> str:
    msgs = [{"role": "system", "content": SYSTEM}]
    if history:
        msgs += history[-6:]
    msgs.append({"role": "user", "content": user})
    resp = _client_get().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msgs,
        temperature=temp,
        max_tokens=700,
    )
    return _clean(resp.choices[0].message.content)


def _factors_text(factors: list[dict]) -> str:
    return "\n".join(
        f"- {f['feature']}: {f['value']} (SHAP {'+' if f['shap'] > 0 else ''}{f['shap']:.3f})"
        for f in factors[:5]
    )


def _summary_text(summary: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in summary.items())


def generate_risk_narrative(prob: float, risk_label: str,
                             top_factors: list[dict], summary: dict) -> str:
    return _chat(f"""Applicant:
{_summary_text(summary)}

Score: {prob:.1%} default probability — {risk_label}

Top factors:
{_factors_text(top_factors)}

Write 3 sentences explaining why this applicant received this score. Be specific. Plain English.""")


def generate_improvement_plan(prob: float, top_factors: list[dict], summary: dict) -> str:
    risk_factors = [f for f in top_factors if f["shap"] > 0]
    return _chat(f"""Applicant default probability: {prob:.1%} — HIGH RISK

Risk factors:
{_factors_text(risk_factors)}

Applicant:
{_summary_text(summary)}

Write 4 numbered, specific, actionable steps to reduce this applicant's default risk. 1-2 sentences each.""")


def generate_protection_tips(prob: float, risk_label: str,
                              top_factors: list[dict], summary: dict) -> str:
    protecting = [f for f in top_factors if f["shap"] < 0][:3]
    risk_drivers = [f for f in top_factors if f["shap"] > 0][:3]
    return _chat(f"""Applicant currently has {prob:.1%} default probability — {risk_label}.

What's protecting them (keep doing these):
{_factors_text(protecting) if protecting else "  None identified."}

What's hurting them (risk drivers):
{_factors_text(risk_drivers)}

Applicant profile:
{_summary_text(summary)}

Write a practical, personalised guide with 2 sections:
1. "What you should keep doing" — 2-3 positive habits already helping this applicant
2. "What to change right now" — 3 concrete steps to immediately reduce default risk
Use bullet points. Be specific to this person's actual numbers and situation. Plain English.""",
    temp=0.45)


def chat_with_analyst(question: str, prob: float, risk_label: str,
                       top_factors: list[dict], summary: dict,
                       history: list[dict]) -> str:
    context = f"""Current applicant: {prob:.1%} default probability — {risk_label}
{_summary_text(summary)}
Top factors:
{_factors_text(top_factors)}"""
    full_system = SYSTEM + f"\n\nApplicant context:\n{context}"
    msgs = [{"role": "system", "content": full_system}]
    msgs += (history or [])[-6:]
    msgs.append({"role": "user", "content": question})
    resp = _client_get().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msgs,
        temperature=0.5,
        max_tokens=600,
    )
    return _clean(resp.choices[0].message.content)
