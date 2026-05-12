import logging
import requests
import os
import json
from .parse_text_agent import AgentState

def _call_azure_openai_for_title_and_summary(text: str) -> tuple[str, str, dict]:
    """Call Azure OpenAI Chat Completions API for title and summary generation
    
    Returns:
        tuple: (title, summary, token_usage_dict)
    """
    
    openai_endpoint = os.environ.get("OPENAI_ENDPOINT")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")
    
    if not openai_endpoint or not openai_api_key:
        logging.warning("Azure OpenAI credentials not found, using fallback title/summary")
        return "", "", {}
    
    # Azure OpenAI Chat Completions endpoint
    chat_completions_url = f"{openai_endpoint}openai/deployments/{openai_model}/chat/completions?api-version=2025-04-01-preview"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": openai_api_key
    }
    
    # Create chat completion messages
    messages = [
        {
            "role": "system",
            "content": "You are a professional document analyst that creates concise titles and summaries for business documents. Return only valid JSON format."
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
        logging.info("Calling Azure OpenAI Chat Completions API for title and summary generation...")
        response = requests.post(chat_completions_url, headers=headers, json=payload, timeout=60)
        
        logging.info(f"Azure OpenAI response status for title/summary: {response.status_code}")
        response.raise_for_status()
        
        result = response.json()
        
        # Extract token usage information
        token_usage = {}
        if "usage" in result:
            token_usage = {
                "prompt_tokens": result["usage"].get("prompt_tokens", 0),
                "completion_tokens": result["usage"].get("completion_tokens", 0),
                "total_tokens": result["usage"].get("total_tokens", 0)
            }
            logging.info(f"Title/Summary Token usage - Prompt: {token_usage['prompt_tokens']}, Completion: {token_usage['completion_tokens']}, Total: {token_usage['total_tokens']}")
        else:
            logging.warning("No usage field found in Azure OpenAI response for title/summary")
        
        # Extract completion text from chat response
        if "choices" in result and len(result["choices"]) > 0:
            completion_text = result["choices"][0]["message"]["content"].strip()
            logging.info(f"Azure OpenAI title/summary response: {completion_text[:200]}...")
            
            # Try to parse as JSON
            try:
                response_data = json.loads(completion_text)
                
                # Extract title and summary
                if isinstance(response_data, dict):
                    title = response_data.get("title", "")
                    summary = response_data.get("summary", "")
                    
                    logging.info(f"Generated title: {title}")
                    logging.info(f"Generated summary: {summary[:100]}...")
                    
                    # Ensure token usage is populated
                    if not token_usage or token_usage == {}:
                        estimated_prompt = len(text.split()) * 1.3
                        estimated_completion = len(completion_text) / 4
                        token_usage = {
                            "prompt_tokens": int(estimated_prompt),
                            "completion_tokens": int(estimated_completion),
                            "total_tokens": int(estimated_prompt + estimated_completion),
                            "estimated": True
                        }
                        logging.warning(f"Token usage was empty, using estimated values: {token_usage}")
                    
                    return title, summary, token_usage
                else:
                    logging.warning("Azure OpenAI returned invalid response format")
                    return "", "", token_usage
                    
            except json.JSONDecodeError as json_err:
                logging.error(f"Failed to parse Azure OpenAI response as JSON: {json_err}")
                logging.error(f"Raw response text: {repr(completion_text)}")
                return "", "", token_usage
        else:
            logging.error("Azure OpenAI response missing choices")
            return "", "", token_usage
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Azure OpenAI API request failed for title/summary: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response content: {e.response.text}")
        return "", "", {}
    except Exception as e:
        logging.error(f"Unexpected error calling Azure OpenAI for title/summary: {str(e)}")
        return "", "", {}

def generate_summary_agent(state: AgentState) -> AgentState:
    """LangGraph node: Generate title and summary using Azure OpenAI"""
    logging.info("="*80)
    logging.info("GENERATE_SUMMARY_AGENT: STARTING")
    logging.info("="*80)
    
    text = state.get("text", "")
    logging.info(f"GENERATE_SUMMARY_AGENT: Input text length: {len(text)} characters")
    logging.info(f"GENERATE_SUMMARY_AGENT: Input state keys: {list(state.keys())}")
    
    logging.info("Starting title and summary generation with Azure OpenAI...")
    
    # Call Azure OpenAI for title and summary
    title, summary, token_usage = _call_azure_openai_for_title_and_summary(text)
    
    logging.info(f"Title/Summary generation result - Title: '{title}', Summary length: {len(summary)} chars")
    logging.info(f"Token usage from API call: {token_usage}")
    
    # Handle fallback if AI didn't return results
    if not title and not summary:
        logging.info("Using fallback title and summary generation")
        # Simple fallback: use first sentence as summary, generate basic title
        sentences = text.split('.')
        title = "Document Analysis"
        summary = sentences[0][:200] + "..." if sentences else "No summary available"
        
        # Estimate token usage for fallback
        if not token_usage or len(token_usage) == 0:
            estimated_prompt = max(len(text.split()) * 1.3, 50)
            estimated_completion = 20
            token_usage = {
                "prompt_tokens": int(estimated_prompt),
                "completion_tokens": estimated_completion,
                "total_tokens": int(estimated_prompt + estimated_completion),
                "estimated": True,
                "reason": "fallback_used"
            }
    
    # Store in state
    state["title"] = title
    state["summary"] = summary
    logging.info(f"GENERATE_SUMMARY_AGENT: Set title in state: '{title}'")
    logging.info(f"GENERATE_SUMMARY_AGENT: Set summary in state: '{summary[:100]}...'")
    
    # Accumulate token usage (add to existing token_usage from synonyms)
    existing_token_usage = state.get("token_usage", {})
    logging.info(f"GENERATE_SUMMARY_AGENT: Existing token_usage from state: {existing_token_usage}")
    
    if existing_token_usage and token_usage:
        # Combine token usage from both API calls
        combined_token_usage = {
            "prompt_tokens": existing_token_usage.get("prompt_tokens", 0) + token_usage.get("prompt_tokens", 0),
            "completion_tokens": existing_token_usage.get("completion_tokens", 0) + token_usage.get("completion_tokens", 0),
            "total_tokens": existing_token_usage.get("total_tokens", 0) + token_usage.get("total_tokens", 0),
        }
        state["token_usage"] = combined_token_usage
        logging.info(f"Combined token usage: {combined_token_usage}")
    elif token_usage:
        state["token_usage"] = token_usage
        logging.info(f"Set token_usage (no existing): {token_usage}")
    
    logging.info("Title and summary generation completed")
    logging.info(f"Final title: {state.get('title', 'MISSING')}")
    logging.info(f"Final summary: {state.get('summary', 'MISSING')[:100]}...")
    logging.info(f"Final token_usage: {state.get('token_usage', 'MISSING')}")
    logging.info("="*80)
    logging.info("GENERATE_SUMMARY_AGENT: COMPLETED")
    logging.info("="*80)
    
    return state
