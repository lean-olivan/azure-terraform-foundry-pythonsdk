import logging
from typing import TypedDict

class AgentState(TypedDict):
    text: str
    parsed_data: dict
    synonyms: dict
    title: str
    summary: str
    enhanced_text: str
    pdf_content: str
    token_usage: dict  # Add token_usage to the state type

def parse_text_agent(state: AgentState) -> AgentState:
    """Parse and analyze input text"""
    text = state.get("text", "")
    
    # Basic text analysis
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    
    parsed_data = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "text_length": len(text),
        "paragraphs": text.split('\n\n')
    }
    
    state["parsed_data"] = parsed_data
    logging.info(f"Parsed text: {word_count} words, {sentence_count} sentences")
    
    return state