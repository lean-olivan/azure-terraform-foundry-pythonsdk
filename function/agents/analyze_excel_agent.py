"""
Excel Analysis Agent
Analyzes Excel content using Azure OpenAI to generate meaningful titles and summaries.
"""

import logging
import os
import json
from typing import TypedDict, Dict, Any, Optional


# ============================================================================
# State Definition
# ============================================================================

class ExcelAgentState(TypedDict):
    """State structure for Excel processing pipeline"""
    text: str
    parsed_data: dict
    synonyms: dict
    title: str
    summary: str
    enhanced_text: str
    pdf_content: str
    token_usage: dict
    excel_content: bytes
    filename: str


# ============================================================================
# Configuration Constants
# ============================================================================

AZURE_API_VERSION = "2024-02-15-preview"
MAX_COMPLETION_TOKENS = 300
TEMPERATURE = 0.3
MAX_TEXT_LENGTH = 2000

SYSTEM_PROMPT = (
    "You are a professional document analyst that creates concise titles "
    "and summaries for Excel files. Return only valid JSON format."
)


# ============================================================================
# Main Agent Function
# ============================================================================

def analyze_excel_agent(state: ExcelAgentState) -> ExcelAgentState:
    """
    Analyze Excel content and generate title and summary using Azure OpenAI.

    Args:
        state: Current pipeline state containing Excel data

    Returns:
        Updated state with title, summary, and token usage
    """

    try:
        # Load Azure OpenAI configuration
        config = _load_openai_config()
        if not config:
            return _set_error_state(
                state,
                "Excel Analysis - Configuration Error",
                "Unable to analyze Excel file due to missing OpenAI configuration."
            )

        # Initialize OpenAI client
        client = _create_openai_client(config)

        # Extract data from state
        parsed_data = state.get("parsed_data", {})
        text = state.get("text", "")
        filename = state.get("filename", "unknown.xlsx")

        # Generate analysis using OpenAI
        analysis_result = _analyze_with_openai(
            client,
            config["model"],
            parsed_data,
            text,
            filename
        )

        # Update state with results
        state["title"] = analysis_result.get("title", "Excel Analysis")
        state["summary"] = analysis_result.get("summary", "Excel file processed successfully")
        state["token_usage"] = analysis_result.get("token_usage", {})

        logging.info(f"Successfully analyzed Excel file '{filename}': {state['title']}")

    except Exception as e:
        logging.error(f"Error analyzing Excel file: {str(e)}")
        return _set_error_state(
            state,
            "Excel Analysis - Error",
            f"Unable to analyze Excel file due to error: {str(e)}"
        )

    return state


# ============================================================================
# Configuration Functions
# ============================================================================

def _load_openai_config() -> Dict[str, str]:
    """
    Load Azure OpenAI configuration from environment variables.

    Returns:
        Dictionary with endpoint, api_key, and model, or None if missing
    """
    endpoint = os.environ.get("OPENAI_ENDPOINT")
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")

    if not endpoint or not api_key:
        logging.error("Missing required environment variables: OPENAI_ENDPOINT or OPENAI_API_KEY")
        return None

    return {
        "endpoint": endpoint,
        "api_key": api_key,
        "model": model
    }


def _create_openai_client(config: Dict[str, str]):
    """
    Create and return an Azure OpenAI client.

    Args:
        config: Dictionary containing endpoint and api_key

    Returns:
        Configured AzureOpenAI client instance
    """
    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=config["endpoint"],
        api_key=config["api_key"],
        api_version=AZURE_API_VERSION
    )


# ============================================================================
# OpenAI Analysis Functions
# ============================================================================

def _analyze_with_openai(
    client,
    model: str,
    parsed_data: Dict[str, Any],
    text: str,
    filename: str
) -> Dict[str, Any]:
    """
    Analyze Excel content using Azure OpenAI API.

    Args:
        client: Azure OpenAI client instance
        model: Model deployment name
        parsed_data: Parsed Excel data structure
        text: Extracted text content from Excel
        filename: Original filename

    Returns:
        Dictionary with title, summary, and token_usage
    """
    # Create the analysis prompt
    prompt = _create_analysis_prompt(parsed_data, text, filename)

    # Call OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"}
    )

    # Parse and return results
    response_text = response.choices[0].message.content
    analysis_result = json.loads(response_text)

    # Add token usage if available
    if hasattr(response, 'usage'):
        analysis_result["token_usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    else:
        analysis_result["token_usage"] = {}

    return analysis_result


def _create_analysis_prompt(
    parsed_data: Dict[str, Any],
    text: str,
    filename: str
) -> str:
    """
    Create a comprehensive analysis prompt for OpenAI.

    Args:
        parsed_data: Parsed Excel data structure
        text: Extracted text content
        filename: Original filename

    Returns:
        Formatted prompt string
    """
    # Extract file information
    file_info = _extract_file_info(parsed_data, filename)

    # Truncate text if too long
    truncated_text = text[:MAX_TEXT_LENGTH]
    if len(text) > MAX_TEXT_LENGTH:
        truncated_text += "..."

    # Build the prompt
    prompt = f"""Analyze this Excel file content and create a professional title and summary.

Excel Content: "{truncated_text}"

File Information:
- Filename: {file_info['filename']}
- Sheets: {file_info['total_sheets']} ({', '.join(file_info['sheet_names'])})
- Size: {file_info['total_rows']} rows × {file_info['total_columns']} columns

Instructions:
- Create a title that captures the essence of the Excel file (5-10 words maximum)
- Create a summary that highlights the main points (2-3 sentences)
- Focus on business/professional context
- Return only valid JSON format

Example JSON format:
{{"title": "Excel File Title Here", "summary": "Excel file summary in 2-3 sentences highlighting the main points and key information."}}

JSON Response:"""

    return prompt


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_file_info(parsed_data: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """
    Extract key file information from parsed data.

    Args:
        parsed_data: Parsed Excel data structure
        filename: Original filename

    Returns:
        Dictionary with file metadata
    """
    return {
        "filename": filename,
        "total_sheets": parsed_data.get("total_sheets", 0),
        "sheet_names": parsed_data.get("sheet_names", []),
        "total_rows": parsed_data.get("total_rows", 0),
        "total_columns": parsed_data.get("total_columns", 0),
        "total_cells": parsed_data.get("total_cells", 0),
        "complexity": parsed_data.get("file_analysis", {}).get("complexity", "Unknown"),
        "data_categories": parsed_data.get("file_analysis", {}).get("data_categories", []),
        "structure_type": parsed_data.get("file_analysis", {}).get("structure_type", "Unknown")
    }


def _set_error_state(state: ExcelAgentState, title: str, summary: str) -> ExcelAgentState:
    """
    Set error information in the state.

    Args:
        state: Current pipeline state
        title: Error title
        summary: Error description

    Returns:
        Updated state with error information
    """
    state["title"] = title
    state["summary"] = summary
    state["token_usage"] = {}
    return state
