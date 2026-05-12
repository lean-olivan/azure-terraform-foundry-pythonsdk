import logging
import requests
import os
import json
from .parse_text_agent import AgentState

def _call_azure_openai_chat_completions_for_synonyms(text: str) -> tuple[dict, dict]:
    """Call Azure OpenAI Chat Completions API for synonym finding
    
    Returns:
        tuple: (synonyms_dict, token_usage_dict)
    """
    
    openai_endpoint = os.environ.get("OPENAI_ENDPOINT")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")
    
    if not openai_endpoint or not openai_api_key:
        logging.warning("Azure OpenAI credentials not found, using fallback synonyms")
        return {}, {}
    
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
            "content": "You are a professional writing assistant that finds synonyms and creates titles and summaries for business documents. Return only valid JSON format."
        },
        {
            "role": "user", 
            "content": f"""Analyze this document text and find professional synonyms for important words, then create a title and summary for the document.

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
        logging.info("Calling Azure OpenAI Chat Completions API for synonym analysis...")
        response = requests.post(chat_completions_url, headers=headers, json=payload, timeout=60)
        
        logging.info(f"Azure OpenAI Chat Completions response status: {response.status_code}")
        response.raise_for_status()
        
        result = response.json()
        
        # Extract token usage information
        print("----------TESTTT----------")
        token_usage = {}
        if "usage" in result:
            token_usage = {
                "prompt_tokens": result["usage"].get("prompt_tokens", 0),
                "completion_tokens": result["usage"].get("completion_tokens", 0),
                "total_tokens": result["usage"].get("total_tokens", 0)
            }
            logging.info(f"Token usage - Prompt: {token_usage['prompt_tokens']}, Completion: {token_usage['completion_tokens']}, Total: {token_usage['total_tokens']}")
        else:
            logging.warning("No usage field found in Azure OpenAI response")
            logging.info(f"Response keys: {list(result.keys())}")
        
        # Extract completion text from chat response
        if "choices" in result and len(result["choices"]) > 0:
            completion_text = result["choices"][0]["message"]["content"].strip()
            logging.info(f"Azure OpenAI chat completion: {completion_text[:200]}...")
            logging.info(f"Raw AI response: {repr(completion_text[:500])}")
            
            # Try to parse as JSON
            try:
                synonyms = json.loads(completion_text)
                
                # Validate it's a dictionary
                if isinstance(synonyms, dict) and synonyms:
                    logging.info(f"Azure OpenAI Chat Completions found synonyms for {len(synonyms)} words")
                    logging.info(f"CRITICAL: Returning AI synonyms with token usage: {token_usage}")
                    logging.info(f"CRITICAL: Token usage type: {type(token_usage)}, empty: {not token_usage}")
                    # Ensure token usage is always populated when AI succeeds
                    if not token_usage or token_usage == {}:
                        # Calculate estimated tokens based on text length
                        estimated_prompt = len(text.split()) * 1.3  # Rough estimate
                        estimated_completion = len(str(synonyms)) / 4  # Rough estimate
                        token_usage = {
                            "prompt_tokens": int(estimated_prompt),
                            "completion_tokens": int(estimated_completion),
                            "total_tokens": int(estimated_prompt + estimated_completion),
                            "estimated": True
                        }
                        logging.warning(f"Token usage was empty, using estimated values: {token_usage}")
                    return synonyms, token_usage
                else:
                    logging.warning("Azure OpenAI returned empty or invalid dictionary")
                    logging.info(f"Returning empty synonyms with token usage: {token_usage}")
                    return {}, token_usage
                    
            except json.JSONDecodeError as json_err:
                logging.error(f"Failed to parse Azure OpenAI completion as JSON: {json_err}")
                logging.error(f"Raw completion text that failed: {repr(completion_text)}")
                logging.info(f"Returning empty synonyms but preserving token usage: {token_usage}")
                return {}, token_usage
        else:
            logging.error("Azure OpenAI response missing choices")
            return {}, token_usage
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Azure OpenAI Chat Completions API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response content: {e.response.text}")
        return {}, {}
    except KeyError as e:
        logging.error(f"Unexpected Azure OpenAI response structure: {str(e)}")
        logging.error(f"Full response: {result}")
        return {}, {}
    except Exception as e:
        logging.error(f"Unexpected error calling Azure OpenAI Chat Completions: {str(e)}")
        return {}, {}

def find_equivalents_agent(state: AgentState) -> AgentState:
    """LangGraph node: Find synonyms using Azure OpenAI Completions with fallback"""
    text = state.get("text", "")
    
    logging.info("Starting synonym analysis with Azure OpenAI Chat Completions...")
    logging.info("LOGING info test line 148")

    # Try Azure OpenAI Chat Completions first
    ai_synonyms, token_usage = _call_azure_openai_chat_completions_for_synonyms(text)
    
    print("- print function return _call_azure_openai_chat_completions_for_synonyms(text) ----------:",_call_azure_openai_chat_completions_for_synonyms(text))
    print("- line 149 ai_synonyms and token_usage ----------:",ai_synonyms, token_usage)
    logging.info("- info logging in line 154 -")

    logging.info(f"AI call result - Synonyms: {len(ai_synonyms) if ai_synonyms else 0} groups, Token usage: {token_usage}")
    
    # ALWAYS estimate token usage if we attempted an AI call (even if it returned no synonyms)
    if not token_usage or len(token_usage) == 0:
        # Estimate based on text length
        estimated_prompt = max(len(text.split()) * 1.3, 50)  # Minimum 50 tokens
        estimated_completion = 30  # Reasonable default
        token_usage = {
            "prompt_tokens": int(estimated_prompt),
            "completion_tokens": estimated_completion,
            "total_tokens": int(estimated_prompt + estimated_completion),
            "estimated": True,
            "reason": "AI_call_attempted_but_no_usage_returned"
        }
        logging.warning(f"Estimated token usage after AI call: {token_usage}")
    
    if ai_synonyms:
        # Use AI-generated synonyms
        state["synonyms"] = ai_synonyms
        state["token_usage"] = token_usage
        logging.info(f"Using Azure OpenAI Chat Completions synonyms for {len(ai_synonyms)} words")
        logging.info(f"Stored token usage in state: {token_usage}")
        logging.info(f"DEBUG: AI synonyms used, token_usage={token_usage}")
        logging.info(f"DEBUG: State keys after AI: {list(state.keys())}")
    else:
        # Fallback to enhanced hardcoded synonyms
        logging.info("Using enhanced fallback synonym mapping")
        
        professional_synonym_map = {
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
            
            # Quality terms
            "good": ["excellent", "superior", "outstanding"],
            "bad": ["poor", "inadequate", "substandard"],
            "big": ["substantial", "significant", "considerable"],
            "small": ["minimal", "limited", "modest"],
            "fast": ["rapid", "swift", "expeditious"],
            "slow": ["gradual", "deliberate", "measured"],
            
            # Action terms
            "show": ["demonstrate", "illustrate", "exhibit"],
            "use": ["utilize", "employ", "apply"],
            "make": ["create", "produce", "generate"],
            "get": ["obtain", "acquire", "secure"],
            "help": ["assist", "support", "facilitate"],
            "work": ["function", "operate", "perform"],
            "find": ["locate", "identify", "discover"],
            "think": ["consider", "contemplate", "analyze"]
        }
        
        # Find synonyms in text
        words = text.lower().split()
        found_synonyms = {}
        
        for word in words:
            # Clean word of punctuation
            clean_word = word.strip('.,!?;:"\'-()[]{}')
            if clean_word in professional_synonym_map:
                found_synonyms[clean_word] = professional_synonym_map[clean_word]
        
        state["synonyms"] = found_synonyms
        # Preserve token usage from AI call (already estimated above)
        state["token_usage"] = token_usage
        logging.info(f"Fallback found synonyms for {len(found_synonyms)} words")
        logging.info(f"Preserved token usage from AI call: {state['token_usage']}")
        logging.info(f"DEBUG: Fallback used, token_usage={state['token_usage']}")
    
    logging.info("Synonym analysis completed")
    logging.info(f"DEBUG: Final state token_usage={state.get('token_usage', 'MISSING')}")
    logging.info(f"DEBUG: Final token_usage type={type(state.get('token_usage'))}, value={state.get('token_usage')}")
    return state