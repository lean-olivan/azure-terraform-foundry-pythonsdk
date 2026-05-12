"""
Summary Generation Agent

Creates professional titles and summaries for documents using Azure OpenAI
Chat Completions API with fallback generation logic.
"""

import logging
import requests
import os
import json
from .parse_text_agent import AgentState


# ==============================================================================
# AZURE OPENAI INTEGRATION
# ==============================================================================

def _call_azure_openai_for_title_and_summary(text: str) -> tuple[str, str, dict]:
    """
    Generate document title and summary using Azure OpenAI Chat Completions API.
    
    Creates concise, professional titles and summaries suitable for
    business documentation and reporting.
    
    Args:
        text: Document text to summarize
        
    Returns:
        Tuple of (title, summary, token_usage_dict)
        - title: Generated document title
        - summary: Generated document summary
        - token_usage_dict: API token consumption metrics
    """
    openai_endpoint = os.environ.get("OPENAI_ENDPOINT")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")
    
    if not openai_endpoint or not openai_api_key:
        logging.warning("Azure OpenAI credentials not configured")
        return "", "", {}
    
    # Construct Chat Completions API endpoint
    api_url = (
        f"{openai_endpoint}openai/deployments/{openai_model}/"
        f"chat/completions?api-version=2025-04-01-preview"
    )
    
    headers = {
        "Content-Type": "application/json",
        "api-key": openai_api_key
    }
    
    # Define conversation for title and summary generation
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional document analyst that creates concise "
                "titles and summaries for business documents. "
                "Return only valid JSON format."
            )
        },
        {
            "role": "user", 
            "content": f"""Analyze this document text and create a professional title and summary.

Document Text: "{text}"

Instructions:
- Create a title that captures the essence of the document (5-10 words maximum)
- Create a summary that highlights the main points (2-3 sentences)
- Focus on business/professional context
- Return only valid JSON format

Example JSON format:
{{"title": "Document Title Here", "summary": "Document summary in 2-3 sentences highlighting the main points and key information."}}

JSON Response:"""
        }
    ]

    payload = {
        "messages": messages,
        "max_completion_tokens": 300,
        "temperature": 0.3,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    try:
        logging.info("Requesting title and summary from Azure OpenAI...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        logging.info(f"Azure OpenAI response status: {response.status_code}")
        response.raise_for_status()
        
        result = response.json()
        
        # Extract token usage metrics
        token_usage = {}
        if "usage" in result:
            token_usage = {
                "prompt_tokens": result["usage"].get("prompt_tokens", 0),
                "completion_tokens": result["usage"].get("completion_tokens", 0),
                "total_tokens": result["usage"].get("total_tokens", 0)
            }
            logging.info(
                f"Token usage - Prompt: {token_usage['prompt_tokens']}, "
                f"Completion: {token_usage['completion_tokens']}, "
                f"Total: {token_usage['total_tokens']}"
            )
        
        # Parse completion text
        if "choices" in result and len(result["choices"]) > 0:
            completion_text = result["choices"][0]["message"]["content"].strip()
            logging.info(f"Received completion: {completion_text[:200]}...")
            
            # Parse JSON response
            try:
                response_data = json.loads(completion_text)
                
                if isinstance(response_data, dict):
                    title = response_data.get("title", "")
                    summary = response_data.get("summary", "")
                    
                    logging.info(f"Generated title: {title}")
                    logging.info(f"Generated summary: {summary[:100]}...")
                    
                    # Ensure token usage is populated
                    if not token_usage:
                        token_usage = _estimate_token_usage(text, completion_text)
                        logging.warning("Token usage estimated from text length")
                    
                    return title, summary, token_usage
                else:
                    logging.warning("Azure OpenAI returned invalid response format")
                    return "", "", token_usage
                    
            except json.JSONDecodeError as json_err:
                logging.error(f"Failed to parse JSON response: {json_err}")
                logging.error(f"Raw completion: {repr(completion_text)}")
                return "", "", token_usage
        else:
            logging.error("Azure OpenAI response missing choices")
            return "", "", token_usage
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Azure OpenAI API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response content: {e.response.text}")
        return "", "", {}
    except Exception as e:
        logging.error(f"Unexpected error calling Azure OpenAI: {str(e)}")
        return "", "", {}


def _estimate_token_usage(input_text: str, output_text: str) -> dict:
    """
    Estimate token usage when actual metrics are unavailable.
    
    Uses rough approximation: 1 token ≈ 0.75 words
    
    Args:
        input_text: Input prompt text
        output_text: Generated completion text
        
    Returns:
        Dictionary with estimated token counts
    """
    estimated_prompt = max(int(len(input_text.split()) * 1.3), 50)
    estimated_completion = max(int(len(output_text) / 4), 10)
    
    return {
        "prompt_tokens": estimated_prompt,
        "completion_tokens": estimated_completion,
        "total_tokens": estimated_prompt + estimated_completion,
        "estimated": True
    }


# ==============================================================================
# FALLBACK GENERATION
# ==============================================================================

def _generate_fallback_title_and_summary(text: str) -> tuple[str, str]:
    """
    Generate basic title and summary when AI is unavailable.
    
    Args:
        text: Document text to process
        
    Returns:
        Tuple of (title, summary)
    """
    title = "Document Analysis"
    
    # Use first sentence as summary
    sentences = text.split('.')
    if sentences:
        summary = sentences[0][:200].strip()
        if len(sentences[0]) > 200:
            summary += "..."
    else:
        summary = "No summary available"
    
    return title, summary


# ==============================================================================
# TOKEN USAGE ACCUMULATION
# ==============================================================================

def _accumulate_token_usage(existing: dict, new: dict) -> dict:
    """
    Combine token usage from multiple API calls.
    
    Args:
        existing: Previously accumulated token usage
        new: New token usage to add
        
    Returns:
        Combined token usage dictionary
    """
    if not existing:
        return new
    
    if not new:
        return existing
    
    return {
        "prompt_tokens": (
            existing.get("prompt_tokens", 0) + 
            new.get("prompt_tokens", 0)
        ),
        "completion_tokens": (
            existing.get("completion_tokens", 0) + 
            new.get("completion_tokens", 0)
        ),
        "total_tokens": (
            existing.get("total_tokens", 0) + 
            new.get("total_tokens", 0)
        ),
    }


# ==============================================================================
# AGENT FUNCTION
# ==============================================================================

def generate_summary_agent(state: AgentState) -> AgentState:
    """
    Generate professional title and summary for the document.
    
    Uses Azure OpenAI to create concise, professional titles and summaries,
    falling back to basic extraction if AI is unavailable. Accumulates token
    usage across multiple API calls in the pipeline.
    
    Args:
        state: Current pipeline state with text to summarize
        
    Returns:
        Updated state with title, summary, and accumulated token_usage
    """
    text = state.get("text", "")
    
    logging.info("Starting title and summary generation...")
    
    # Attempt AI-powered generation
    title, summary, token_usage = _call_azure_openai_for_title_and_summary(text)
    
    # Use fallback if AI didn't return results
    if not title and not summary:
        logging.info("Using fallback title and summary generation")
        title, summary = _generate_fallback_title_and_summary(text)
        
        # Estimate token usage for fallback
        if not token_usage:
            token_usage = _estimate_token_usage(text, title + summary)
            token_usage["reason"] = "fallback_used"
    
    # Store results in state
    state["title"] = title
    state["summary"] = summary
    
    logging.info(f"Title: {title}")
    logging.info(f"Summary: {summary[:100]}...")
    
    # Accumulate token usage from previous agents
    existing_token_usage = state.get("token_usage", {})
    state["token_usage"] = _accumulate_token_usage(existing_token_usage, token_usage)
    
    logging.info(f"Accumulated token usage: {state['token_usage']}")
    logging.info("Title and summary generation completed")
    
    return state
