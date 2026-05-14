"""
Agents Package

This package contains specialized agent workflows organized by file type:
- text: Agents for processing text and Word documents
- excel: Agents for processing Excel files

Each subdirectory contains its own pipeline orchestration and specialized agents.
"""

# Re-export text processing components
from .text import (
    parse_text_agent,
    AgentState,
    find_equivalents_agent,
    generate_summary_agent,
    consolidate_text_agent,
    run_langgraph_document_pipeline,
)

# Re-export Excel processing components
from .excel import (
    parse_excel_agent,
    ExcelAgentState,
    analyze_excel_agent,
    run_excel_langgraph_pipeline,
)

__all__ = [
    # Text processing
    'parse_text_agent',
    'AgentState',
    'find_equivalents_agent',
    'generate_summary_agent',
    'consolidate_text_agent',
    'run_langgraph_document_pipeline',
    # Excel processing
    'parse_excel_agent',
    'ExcelAgentState',
    'analyze_excel_agent',
    'run_excel_langgraph_pipeline',
]
