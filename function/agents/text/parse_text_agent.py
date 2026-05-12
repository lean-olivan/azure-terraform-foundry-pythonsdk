"""
Text Parser Agent

Analyzes and extracts metadata from text documents, providing basic statistics
and structure information for downstream processing.
"""

import logging
from typing import TypedDict


# ==============================================================================
# STATE DEFINITION
# ==============================================================================

class AgentState(TypedDict):
    """
    State structure shared across all text processing agents.
    
    Attributes:
        text: Original document text content
        parsed_data: Extracted metadata and statistics
        synonyms: Dictionary mapping words to their synonyms
        title: Generated document title
        summary: Generated document summary
        enhanced_text: Text with applied enhancements
        pdf_content: Base64-encoded PDF output
        token_usage: API token consumption statistics
    """
    text: str
    parsed_data: dict
    synonyms: dict
    title: str
    summary: str
    enhanced_text: str
    pdf_content: str
    token_usage: dict


# ==============================================================================
# AGENT FUNCTION
# ==============================================================================

def parse_text_agent(state: AgentState) -> AgentState:
    """
    Parse and analyze input text to extract basic statistics.
    
    This agent performs fundamental text analysis including word count,
    sentence count, and paragraph segmentation. The results are stored
    in the state for use by subsequent agents in the pipeline.
    
    Args:
        state: Current pipeline state containing text to analyze
        
    Returns:
        Updated state with parsed_data populated
    """
    text = state.get("text", "")
    
    # Calculate basic text metrics
    words = text.split()
    word_count = len(words)
    
    # Count sentences by punctuation markers
    sentence_count = (
        text.count('.') + 
        text.count('!') + 
        text.count('?')
    )
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    # Store analysis results
    parsed_data = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "text_length": len(text),
        "paragraphs": paragraphs
    }
    
    state["parsed_data"] = parsed_data
    
    logging.info(
        f"Text parsing complete: {word_count} words, "
        f"{sentence_count} sentences, {len(paragraphs)} paragraphs"
    )
    
    return state
