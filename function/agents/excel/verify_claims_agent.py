"""
Claim-Level Verification Agent

Verifies every factual claim made in the LLM-generated title and summary
against the original source document (Excel extracted text + ASC 805 reference).

Uses an LLM-as-judge approach: a second Azure OpenAI call checks whether each
distinct claim is:
  - "explicit"   – directly stated in the source
  - "inferred"   – a reasonable inference from the source
  - "unsupported" – invented / hallucinated

The overall_faithfulness score (0.0–1.0) is computed as:
    (explicit + inferred) / total_claims
"""

import logging
import os
import json
from typing import Dict, Any

from .parse_excel_agent import ExcelAgentState


# ============================================================================
# Configuration
# ============================================================================

AZURE_API_VERSION = "2024-02-15-preview"
MAX_COMPLETION_TOKENS = 1500
TEMPERATURE = 0.1          # Very low – factual verification needs determinism
MAX_SOURCE_CHARS = 8000    # Limit source text sent to avoid token overflow


# ============================================================================
# Main Agent Function
# ============================================================================

def verify_claims_agent(state: ExcelAgentState) -> ExcelAgentState:
    """
    Verify factual claims in the generated title and summary against source.

    Uses Azure OpenAI as an LLM judge to evaluate whether each claim in the
    generated analysis is explicitly stated, reasonably inferred, or
    unsupported (hallucinated) relative to the source document.

    Args:
        state: Current pipeline state with text, title, and summary

    Returns:
        Updated state with claim_verification populated
    """
    filename = state.get("filename", "unknown.xlsx")
    logging.info(f"Starting Claim-Level Verification for: {filename}")

    title = state.get("title", "")
    summary = state.get("summary", "")

    # Nothing to verify if no analysis was generated
    if not title and not summary:
        logging.warning("No title or summary available for claim verification")
        state["claim_verification"] = {
            "error": "No title or summary to verify",
            "verified": False
        }
        return state

    # Check Azure OpenAI configuration
    endpoint = os.environ.get("OPENAI_ENDPOINT")
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")

    if not endpoint or not api_key:
        logging.error("Azure OpenAI credentials not configured for claim verification")
        state["claim_verification"] = {
            "error": "Azure OpenAI credentials not configured",
            "verified": False
        }
        return state

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=AZURE_API_VERSION
        )

        # Build source context from extracted text + ASC 805 reference document
        source_context = _build_source_context(state)

        # Run claim verification for title
        title_result = _verify_claims(
            client, model, source_context,
            generated_text=title,
            claim_type="title"
        )

        # Run claim verification for summary
        summary_result = _verify_claims(
            client, model, source_context,
            generated_text=summary,
            claim_type="summary"
        )

        # Compute overall scores across both title and summary claims
        all_claims = (
            title_result.get("claims", []) +
            summary_result.get("claims", [])
        )
        overall_faithfulness = _compute_faithfulness_score(all_claims)
        overall_confidence = _compute_overall_confidence(all_claims)

        # Accumulate token usage from both verification API calls
        title_usage = title_result.pop("token_usage", {})
        summary_usage = summary_result.pop("token_usage", {})
        verification_usage = _merge_token_usage(title_usage, summary_usage)
        existing_usage = state.get("token_usage", {})
        state["token_usage"] = _merge_token_usage(existing_usage, verification_usage)

        state["claim_verification"] = {
            "title_verification": title_result,
            "summary_verification": summary_result,
            "overall_faithfulness": overall_faithfulness,
            "overall_confidence": overall_confidence,
            "total_claims": len(all_claims),
            "verified": True
        }

        logging.info(
            f"Claim verification completed. "
            f"Overall faithfulness: {overall_faithfulness:.2f}, "
            f"Overall confidence: {overall_confidence:.2f} "
            f"({len(all_claims)} claims checked)"
        )

    except Exception as e:
        logging.error(f"Claim verification failed: {str(e)}", exc_info=True)
        state["claim_verification"] = {
            "error": f"Claim verification failed: {str(e)}",
            "verified": False
        }

    return state


# ============================================================================
# Core Verification Logic
# ============================================================================

def _verify_claims(
    client,
    model: str,
    source_context: str,
    generated_text: str,
    claim_type: str
) -> Dict[str, Any]:
    """
    Ask the LLM to verify every claim in generated_text against source_context.

    Args:
        client: AzureOpenAI client
        model: Deployment name
        source_context: Source document text used as ground truth
        generated_text: LLM-generated text whose claims are to be verified
        claim_type: Label for logging ("title" or "summary")

    Returns:
        Dictionary with claims list and overall_faithfulness for this text
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a fact-checking auditor evaluating whether an AI-generated "
                "title or summary is grounded in the source document. "
                "Apply the following rules strictly:\n"
                "  'explicit'   – The claim (or its clear paraphrase/title-case variant) "
                "is directly present in the source. Minor spelling differences, "
                "title-casing, or word reordering still count as explicit.\n"
                "  'inferred'   – The claim is a reasonable, unambiguous conclusion "
                "drawn from information present in the source (e.g. labelling content "
                "as a 'template' when the source describes a workbook designed for "
                "testing and documentation).\n"
                "  'unsupported' – There is no basis in the source for the claim. "
                "Use this only when the claim introduces concepts not present anywhere "
                "in the source.\n"
                "Be generous with 'explicit' and 'inferred' for titles and summaries "
                "since they are by nature condensed paraphrases of source content. "
                "Reserve 'unsupported' for genuine hallucinations. "
                "Return ONLY valid JSON – no prose, no markdown."
            )
        },
        {
            "role": "user",
            "content": f"""Verify whether the generated {claim_type} is grounded in the source document.

Source Document:
{source_context}

Generated {claim_type.title()}:
{generated_text}

For every distinct factual claim or named concept in the generated {claim_type}, produce an entry with:
- "claim": the exact phrase from the generated text
- "status": "explicit" | "inferred" | "unsupported"  (see rules above)
- "evidence": the closest matching text from the source, or "none"
- "confidence": your confidence that the status is correct (0.0–1.0)

Then compute overall_faithfulness = (explicit_count + inferred_count) / total_claims.
If there are no claims, set overall_faithfulness to 1.0.

Return ONLY this JSON:
{{
    "claims": [
        {{
            "claim": "...",
            "status": "explicit|inferred|unsupported",
            "evidence": "...",
            "confidence": 0.95
        }}
    ],
    "overall_faithfulness": 1.0,
    "explicit_count": 0,
    "inferred_count": 0,
    "unsupported_count": 0
}}

JSON Response:"""
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=TEMPERATURE,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content.strip()
        result = json.loads(response_text)

        # Attach token usage so the caller can accumulate it into state
        if hasattr(response, "usage") and response.usage:
            result["token_usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        logging.info(
            f"Claim verification for {claim_type}: "
            f"faithfulness={result.get('overall_faithfulness', 'N/A')}, "
            f"claims={len(result.get('claims', []))}"
        )

        return result

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse claim verification JSON for {claim_type}: {e}")
        return {"error": f"JSON parse error: {str(e)}", "claims": [], "overall_faithfulness": None}
    except Exception as e:
        logging.error(f"Claim verification API call failed for {claim_type}: {e}")
        return {"error": str(e), "claims": [], "overall_faithfulness": None}


# ============================================================================
# Helper Functions
# ============================================================================

def _build_source_context(state: ExcelAgentState) -> str:
    """
    Combine all available Excel metadata and content with the ASC 805 reference
    document to form the ground-truth source context for claim verification.

    Includes:
    - Filename and sheet names (title claims often derive from these)
    - File structure metadata (complexity, categories, structure type)
    - Extracted cell content
    - ASC 805 reference document

    Truncates to MAX_SOURCE_CHARS to stay within model token limits.

    Args:
        state: Current pipeline state

    Returns:
        Combined source context string
    """
    # Import here to avoid circular import; module-level constant already loaded
    from .analyze_excel_agent import ASC_805_REFERENCE_CONTEXT

    extracted_text = state.get("text", "")
    filename = state.get("filename", "unknown.xlsx")
    parsed_data = state.get("parsed_data", {})

    parts = []

    # ── 1. File metadata (sheet names, structure) ──────────────────────────────
    # This is critical: the title is often derived from the filename and sheet
    # names, which are NOT in the raw cell text extracted by parse_excel_agent.
    sheet_names = parsed_data.get("sheet_names", [])
    total_sheets = parsed_data.get("total_sheets", 0)
    total_rows = parsed_data.get("total_rows", 0)
    total_columns = parsed_data.get("total_columns", 0)
    file_analysis = parsed_data.get("file_analysis", {})
    complexity = file_analysis.get("complexity", "")
    categories = file_analysis.get("data_categories", [])
    structure_type = file_analysis.get("structure_type", "")
    content_summary = file_analysis.get("content_summary", "")

    metadata_lines = [f"Excel File Metadata: {filename}"]
    if sheet_names:
        metadata_lines.append(f"Sheet Names: {', '.join(sheet_names)}")
    if total_sheets:
        metadata_lines.append(f"Total Sheets: {total_sheets}")
    if total_rows or total_columns:
        metadata_lines.append(f"Size: {total_rows} rows × {total_columns} columns")
    if complexity:
        metadata_lines.append(f"Complexity: {complexity}")
    if categories:
        metadata_lines.append(f"Data Categories: {', '.join(categories)}")
    if structure_type:
        metadata_lines.append(f"Structure Type: {structure_type}")
    if content_summary:
        metadata_lines.append(f"Content Summary: {content_summary}")

    parts.append("\n".join(metadata_lines))

    # ── 2. Extracted cell text ─────────────────────────────────────────────────
    if extracted_text:
        parts.append(f"Extracted Cell Content:\n{extracted_text}")

    # ── 3. ASC 805 reference document ─────────────────────────────────────────
    if ASC_805_REFERENCE_CONTEXT:
        parts.append(f"Reference Standard (ASC 805):\n{ASC_805_REFERENCE_CONTEXT}")

    source = "\n\n---\n\n".join(parts)

    # Truncate to prevent token overflow (metadata is always kept; truncation
    # hits the cell content and reference doc tail if needed)
    if len(source) > MAX_SOURCE_CHARS:
        source = source[:MAX_SOURCE_CHARS] + "\n...[truncated]"
        logging.debug(f"Source context truncated to {MAX_SOURCE_CHARS} chars")

    return source


def _merge_token_usage(a: dict, b: dict) -> dict:
    """
    Merge two token-usage dicts by summing their counters.

    Args:
        a: First token usage dict (may be empty)
        b: Second token usage dict (may be empty)

    Returns:
        Combined token usage dict
    """
    if not a:
        return b
    if not b:
        return a
    return {
        "prompt_tokens": a.get("prompt_tokens", 0) + b.get("prompt_tokens", 0),
        "completion_tokens": a.get("completion_tokens", 0) + b.get("completion_tokens", 0),
        "total_tokens": a.get("total_tokens", 0) + b.get("total_tokens", 0),
    }


def _compute_faithfulness_score(claims: list) -> float:
    """
    Compute overall faithfulness score from a list of claim results.

    Score = (explicit + inferred) / total
    Returns 1.0 if there are no claims (nothing to hallucinate).

    Args:
        claims: List of claim dicts with "status" field

    Returns:
        Faithfulness score between 0.0 and 1.0
    """
    if not claims:
        return 1.0

    supported = sum(
        1 for c in claims
        if c.get("status") in ("explicit", "inferred")
    )

    return round(supported / len(claims), 4)


def _compute_overall_confidence(claims: list) -> float:
    """
    Compute the average confidence across all claims.

    This measures how certain the LLM judge was across all its verdicts,
    regardless of whether each claim passed or failed. A high overall_confidence
    means the judge was decisive; a low value means the source was ambiguous.

    Args:
        claims: List of claim dicts with "confidence" field (0.0–1.0)

    Returns:
        Average confidence score between 0.0 and 1.0,
        or 1.0 if there are no claims.
    """
    if not claims:
        return 1.0

    confidences = [
        float(c.get("confidence", 1.0))
        for c in claims
        if c.get("confidence") is not None
    ]

    if not confidences:
        return 1.0

    return round(sum(confidences) / len(confidences), 4)
