"""
Synonym Finding Agent

Identifies professional synonyms for key terms in documents using Azure OpenAI
Chat Completions API with intelligent fallback mechanisms.
"""

import logging
import requests
import os
import json
from .parse_text_agent import AgentState


# ==============================================================================
# AZURE OPENAI INTEGRATION
# ==============================================================================

def _call_azure_openai_for_synonyms(text: str) -> tuple[dict, dict]:
    """
    Request synonym suggestions from Azure OpenAI Chat Completions API.
    
    Uses a structured prompt to identify important words and generate
    professional synonyms suitable for business documents.
    
    Args:
        text: Document text to analyze
        
    Returns:
        Tuple of (synonyms_dict, token_usage_dict)
        - synonyms_dict: Maps words to lists of synonym alternatives
        - token_usage_dict: API token consumption metrics
    """
    openai_endpoint = os.environ.get("OPENAI_ENDPOINT")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")
    
    if not openai_endpoint or not openai_api_key:
        logging.warning("Azure OpenAI credentials not configured")
        return {}, {}
    
    # Construct Chat Completions API endpoint
    api_url = (
        f"{openai_endpoint}openai/deployments/{openai_model}/"
        f"chat/completions?api-version=2025-04-01-preview"
    )
    
    headers = {
        "Content-Type": "application/json",
        "api-key": openai_api_key
    }
    
    # Define conversation for synonym extraction
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional writing assistant that finds synonyms "
                "and creates titles and summaries for business documents. "
                "Return only valid JSON format."
            )
        },
        {
            "role": "user", 
            "content": f"""Analyze this document text and find professional synonyms for important words.

Document Text: "{text}"

Instructions:
- Find nouns, verbs, and adjectives that can be enhanced
- Skip common words (the, and, is, a, an, to, for, etc.)
- Provide 3 professional synonyms for each word
- Return only valid JSON format
- Focus on business/professional context

Example JSON format:
{{"analyze": ["examine", "evaluate", "assess"], "important": ["crucial", "vital", "significant"]}}

JSON Response:"""
        }
    ]

    payload = {
        "messages": messages,
        "max_completion_tokens": 800,
        "temperature": 0.2,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    try:
        logging.info("Requesting synonyms from Azure OpenAI...")
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
                synonyms = json.loads(completion_text)
                
                if isinstance(synonyms, dict) and synonyms:
                    logging.info(f"Found synonyms for {len(synonyms)} words")
                    
                    # Ensure token usage is populated
                    if not token_usage:
                        token_usage = _estimate_token_usage(text, str(synonyms))
                        logging.warning("Token usage estimated from text length")
                    
                    return synonyms, token_usage
                else:
                    logging.warning("Azure OpenAI returned empty dictionary")
                    return {}, token_usage
                    
            except json.JSONDecodeError as json_err:
                logging.error(f"Failed to parse JSON response: {json_err}")
                logging.error(f"Raw completion: {repr(completion_text)}")
                return {}, token_usage
        else:
            logging.error("Azure OpenAI response missing choices")
            return {}, token_usage
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Azure OpenAI API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response content: {e.response.text}")
        return {}, {}
    except Exception as e:
        logging.error(f"Unexpected error calling Azure OpenAI: {str(e)}")
        return {}, {}


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
# FALLBACK SYNONYM MAPPING
# ==============================================================================

def _get_fallback_synonyms() -> dict:
    """
    Provide hardcoded professional synonym mappings as fallback.
    
    Returns:
        Dictionary mapping common words to professional alternatives
    """
    return {
        # Business terms
        "analyze": ["examine", "evaluate", "assess"],
        "important": ["crucial", "vital", "significant"],
        "document": ["report", "manuscript", "file"],
        "process": ["procedure", "method", "workflow"],
        "review": ["examine", "evaluate", "inspect"],
        "implement": ["execute", "deploy", "establish"],
        "manage": ["oversee", "coordinate", "supervise"],
        "develop": ["create", "establish", "formulate"],
        "improve": ["enhance", "optimize", "refine"],
        "effective": ["efficient", "successful", "productive"],
        
        # Quality descriptors
        "good": ["excellent", "superior", "outstanding"],
        "bad": ["poor", "inadequate", "substandard"],
        "big": ["substantial", "significant", "considerable"],
        "small": ["minimal", "limited", "modest"],
        "fast": ["rapid", "swift", "expeditious"],
        "slow": ["gradual", "deliberate", "measured"],
        
        # Action verbs
        "show": ["demonstrate", "illustrate", "exhibit"],
        "use": ["utilize", "employ", "apply"],
        "make": ["create", "produce", "generate"],
        "get": ["obtain", "acquire", "secure"],
        "help": ["assist", "support", "facilitate"],
        "work": ["function", "operate", "perform"],
        "find": ["locate", "identify", "discover"],
        "think": ["consider", "contemplate", "analyze"]
    }


def _find_synonyms_in_text(text: str, synonym_map: dict) -> dict:
    """
    Search text for words that have synonym mappings.
    
    Args:
        text: Document text to search
        synonym_map: Dictionary of word-to-synonyms mappings
        
    Returns:
        Dictionary of synonyms found in the text
    """
    words = text.lower().split()
    found_synonyms = {}
    
    for word in words:
        # Remove punctuation
        clean_word = word.strip('.,!?;:"\'-()[]{}')
        
        if clean_word in synonym_map:
            found_synonyms[clean_word] = synonym_map[clean_word]
    
    return found_synonyms


# ==============================================================================
# AGENT FUNCTION
# ==============================================================================

def find_equivalents_agent(state: AgentState) -> AgentState:
    """
    Find professional synonyms for key terms using AI or fallback mapping.
    
    Attempts to use Azure OpenAI for intelligent synonym detection, falling
    back to a curated professional synonym list if AI is unavailable or fails.
    
    Args:
        state: Current pipeline state with text to analyze
        
    Returns:
        Updated state with synonyms and token_usage populated
    """
    text = state.get("text", "")
    
    logging.info("Starting synonym analysis...")

    # Attempt AI-powered synonym detection
    ai_synonyms, token_usage = _call_azure_openai_for_synonyms(text)
    
    # Ensure token usage is estimated if AI was attempted but returned nothing
    if not token_usage:
        token_usage = _estimate_token_usage(text, "")
        token_usage["reason"] = "AI_call_attempted_but_no_usage_returned"
        logging.warning(f"Estimated token usage: {token_usage}")
    
    if ai_synonyms:
        # Success: Use AI-generated synonyms
        state["synonyms"] = ai_synonyms
        state["token_usage"] = token_usage
        logging.info(f"Using AI synonyms for {len(ai_synonyms)} words")
    else:
        # Fallback: Use curated professional synonym list
        logging.info("Using fallback professional synonym mapping")
        
        synonym_map = _get_fallback_synonyms()
        found_synonyms = _find_synonyms_in_text(text, synonym_map)
        
        state["synonyms"] = found_synonyms
        state["token_usage"] = token_usage
        
        logging.info(f"Fallback found synonyms for {len(found_synonyms)} words")
    
    logging.info("Synonym analysis completed")
    return state
