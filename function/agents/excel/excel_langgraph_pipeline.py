"""
Excel LangGraph Pipeline
Orchestrates the Excel processing workflow using LangGraph framework.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any
from langgraph.graph import StateGraph, END

# Import Excel-specific agents
from .parse_excel_agent import parse_excel_agent, ExcelAgentState
from .analyze_excel_agent import analyze_excel_agent
from .evaluate_ragas_agent import evaluate_ragas_agent


# ============================================================================
# Constants
# ============================================================================

PIPELINE_VERSION = "1.0"
WORKFLOW_TYPE = "excel_langgraph_pipeline"


# ============================================================================
# Main Pipeline Function
# ============================================================================

def run_excel_langgraph_pipeline(
    excel_content: bytes, 
    filename: str = "unknown.xlsx"
) -> Dict[str, Any]:
    """
    Run the complete Excel processing pipeline using LangGraph.
    
    This pipeline orchestrates the Excel file processing workflow through
    multiple specialized agents:
    1. parse_excel: Extract structured data and text from Excel file
    2. analyze_excel: Use Azure OpenAI to generate title and summary
    
    Args:
        excel_content: Raw bytes of the Excel file
        filename: Name of the Excel file
        
    Returns:
        Dictionary containing the complete analysis results with metadata
    """
    logging.info(f"Starting Excel LangGraph pipeline for: {filename}")
    
    try:
        # Initialize the pipeline state
        initial_state = _create_initial_state(excel_content, filename)
        
        # Create and execute the workflow
        workflow = create_excel_workflow()
        result = workflow.invoke(initial_state)
        
        # Prepare final response
        final_result = _prepare_success_response(result, filename)
        
        logging.info(f"Excel LangGraph pipeline completed successfully for: {filename}")
        return final_result
        
    except Exception as e:
        logging.error(f"Excel LangGraph pipeline failed for {filename}: {str(e)}")
        return _prepare_error_response(e, filename)


# ============================================================================
# Workflow Creation
# ============================================================================

def create_excel_workflow() -> StateGraph:
    """
    Create the Excel processing workflow graph using LangGraph.
    
    Defines the workflow structure:
    - Entry Point: parse_excel (extract data from Excel)
    - Next Node: analyze_excel (generate AI analysis)
    - Next Node: evaluate_ragas (evaluate quality with RAGAS metrics)
    - End: Return final results
    
    Returns:
        Compiled LangGraph workflow ready for execution
    """
    # Initialize the workflow with state definition
    workflow = StateGraph(ExcelAgentState)
    
    # Add processing nodes (agents)
    workflow.add_node("parse_excel", parse_excel_agent)
    workflow.add_node("analyze_excel", analyze_excel_agent)
    workflow.add_node("evaluate_ragas", evaluate_ragas_agent)
    
    # Define the execution flow
    workflow.set_entry_point("parse_excel")
    workflow.add_edge("parse_excel", "analyze_excel")
    workflow.add_edge("analyze_excel", "evaluate_ragas")
    workflow.add_edge("evaluate_ragas", END)
    
    # Compile and return the workflow
    return workflow.compile()


# ============================================================================
# State Management Functions
# ============================================================================

def _create_initial_state(excel_content: bytes, filename: str) -> ExcelAgentState:
    """
    Create the initial state for the pipeline.
    
    Args:
        excel_content: Raw Excel file bytes
        filename: Name of the file
        
    Returns:
        Initial ExcelAgentState with empty fields
    """
    return ExcelAgentState(
        text="",
        parsed_data={},
        synonyms={},
        title="",
        summary="",
        enhanced_text="",
        pdf_content="",
        token_usage={},
        excel_content=excel_content,
        filename=filename,
        ragas_scores={}
    )


# ============================================================================
# Response Preparation Functions
# ============================================================================

def _prepare_success_response(result: ExcelAgentState, filename: str) -> Dict[str, Any]:
    """
    Prepare a successful pipeline response.
    
    Args:
        result: Final state after pipeline execution
        filename: Name of the processed file
        
    Returns:
        Dictionary with standardized success response
    """
    return {
        "success": True,
        "filename": filename,
        "title": result.get("title", ""),
        "summary": result.get("summary", ""),
        "parsed_data": result.get("parsed_data", {}),
        "token_usage": result.get("token_usage", {}),
        "ragas_scores": result.get("ragas_scores", {}),
        "workflow_info": {
            "agents_used": ["parse_excel", "analyze_excel", "evaluate_ragas"],
            "workflow_type": WORKFLOW_TYPE,
            "version": PIPELINE_VERSION,
            "langgraph_enabled": True,
            "azure_openai_enabled": True,
            "ragas_enabled": True,
            "processing_timestamp": _get_current_timestamp()
        }
    }


def _prepare_error_response(error: Exception, filename: str) -> Dict[str, Any]:
    """
    Prepare an error response when pipeline fails.
    
    Args:
        error: Exception that caused the failure
        filename: Name of the file that failed
        
    Returns:
        Dictionary with standardized error response
    """
    error_message = str(error)
    
    return {
        "success": False,
        "filename": filename,
        "error": error_message,
        "title": "Excel Processing - Error",
        "summary": f"Failed to process Excel file: {error_message}",
        "parsed_data": {"error": error_message},
        "token_usage": {},
        "workflow_info": {
            "agents_used": ["parse_excel", "analyze_excel"],
            "workflow_type": WORKFLOW_TYPE,
            "version": PIPELINE_VERSION,
            "langgraph_enabled": True,
            "azure_openai_enabled": True,
            "error_timestamp": _get_current_timestamp()
        }
    }


# ============================================================================
# Utility Functions
# ============================================================================

def _get_current_timestamp() -> str:
    """
    Get current timestamp in ISO 8601 format with UTC timezone.
    
    Returns:
        ISO formatted timestamp string
    """
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# Backward Compatibility
# ============================================================================

def run_langgraph_excel_pipeline(
    excel_content: bytes, 
    filename: str = "unknown.xlsx"
) -> Dict[str, Any]:
    """
    Alias for run_excel_langgraph_pipeline.
    
    Provided for backward compatibility with existing code that may use
    the alternative function name.
    
    Args:
        excel_content: Raw bytes of the Excel file
        filename: Name of the Excel file
        
    Returns:
        Dictionary containing the complete analysis results
    """
    return run_excel_langgraph_pipeline(excel_content, filename)
