"""
LangGraph Text Processing Pipeline

Orchestrates the sequential execution of text processing agents using LangGraph's
state machine framework for reliable, maintainable workflow management.
"""

from langgraph.graph import StateGraph, START, END
import logging

from .parse_text_agent import parse_text_agent, AgentState
from .find_equivalents_agent import find_equivalents_agent
from .generate_summary_agent import generate_summary_agent
from .consolidate_text_agent import consolidate_text_agent


# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================

def create_document_processing_graph():
    """
    Create and configure the LangGraph state machine for document processing.
    
    Pipeline Flow:
        1. parse_text: Extract document statistics and structure
        2. find_equivalents: Identify professional synonyms using AI
        3. generate_summary: Create title and summary using AI
        4. consolidate_text: Apply enhancements and generate PDF
        
    Returns:
        Compiled LangGraph application ready for execution
    """
    # Initialize state graph with AgentState type
    graph = StateGraph(AgentState)
    
    # Register all agent nodes
    graph.add_node("parse_text", parse_text_agent)
    graph.add_node("find_equivalents", find_equivalents_agent)
    graph.add_node("generate_summary", generate_summary_agent)
    graph.add_node("consolidate_text", consolidate_text_agent)
    
    # Define sequential execution flow
    graph.add_edge(START, "parse_text")
    graph.add_edge("parse_text", "find_equivalents")
    graph.add_edge("find_equivalents", "generate_summary")
    graph.add_edge("generate_summary", "consolidate_text")
    graph.add_edge("consolidate_text", END)
    
    # Compile the graph for execution
    compiled_graph = graph.compile()
    
    logging.info("LangGraph document processing pipeline created")
    return compiled_graph


# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================

def run_langgraph_document_pipeline(text: str, filename: str = "document.docx") -> dict:
    """
    Execute the complete document processing pipeline.
    
    Processes text through all agents in sequence, accumulating state
    and results along the way. Handles Azure OpenAI integration for
    AI-powered enhancements.
    
    Args:
        text: Document text to process
        filename: Name of the document file (for logging)
        
    Returns:
        Final state dictionary containing all processing results:
        - text: Original input text
        - parsed_data: Document statistics
        - synonyms: Found synonym mappings
        - title: Generated document title
        - summary: Generated document summary
        - enhanced_text: Text with applied synonyms
        - pdf_content: Base64-encoded comparison PDF
        - token_usage: Accumulated API token usage
    """
    # Initialize pipeline state
    initial_state = {
        "text": text,
        "parsed_data": {},
        "synonyms": {},
        "title": "",
        "summary": "",
        "enhanced_text": "",
        "pdf_content": "",
        "token_usage": {},
    }
    
    # Create pipeline instance
    pipeline = create_document_processing_graph()
    
    logging.info(f"Starting LangGraph pipeline for: {filename}")
    logging.info(f"Input text length: {len(text)} characters")
    
    # Execute pipeline
    final_state = pipeline.invoke(initial_state)
    
    # Log results
    logging.info(f"Pipeline completed for: {filename}")
    logging.info(f"Generated title: {final_state.get('title', 'N/A')}")
    logging.info(f"Summary length: {len(final_state.get('summary', ''))} characters")
    logging.info(f"Synonyms found: {len(final_state.get('synonyms', {}))}")
    logging.info(f"Token usage: {final_state.get('token_usage', {})}")
    
    return final_state
