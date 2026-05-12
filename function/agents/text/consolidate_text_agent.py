"""
Text Consolidation Agent

Applies synonym enhancements to text and generates PDF output with
comparison of original and enhanced versions.
"""

import logging
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from .parse_text_agent import AgentState


# ==============================================================================
# TEXT ENHANCEMENT
# ==============================================================================

def _apply_synonym_enhancements(text: str, synonyms: dict) -> str:
    """
    Replace words in text with their preferred synonyms.
    
    Handles multiple synonym formats (list, dict, string) for flexibility
    with different AI response structures.
    
    Args:
        text: Original document text
        synonyms: Dictionary mapping words to synonyms
        
    Returns:
        Enhanced text with synonyms applied
    """
    enhanced_text = text
    
    for word, synonym_data in synonyms.items():
        replacement = None
        
        # Handle list format: ["synonym1", "synonym2", "synonym3"]
        if isinstance(synonym_data, list) and synonym_data:
            replacement = synonym_data[0]
        
        # Handle dict format: {"1": "synonym1", "2": "synonym2"}
        elif isinstance(synonym_data, dict):
            first_value = next(iter(synonym_data.values()), None)
            if first_value and isinstance(first_value, str):
                replacement = first_value
            else:
                logging.warning(
                    f"Dict synonym format has no usable value for '{word}': "
                    f"{synonym_data}"
                )
        
        # Handle string format: "synonym"
        elif isinstance(synonym_data, str):
            replacement = synonym_data
        
        else:
            logging.warning(
                f"Unexpected synonym format for '{word}': "
                f"{type(synonym_data)}"
            )
        
        # Apply replacement if found
        if replacement:
            enhanced_text = enhanced_text.replace(word, replacement)
    
    return enhanced_text


# ==============================================================================
# PDF GENERATION
# ==============================================================================

def _generate_comparison_pdf(original_text: str, enhanced_text: str) -> str:
    """
    Create a PDF comparing original and enhanced text versions.
    
    Args:
        original_text: Original document text
        enhanced_text: Enhanced text with applied synonyms
        
    Returns:
        Base64-encoded PDF content, or empty string on error
    """
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Document title
        title = Paragraph("Enhanced Text Document", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Original text section
        original_heading = Paragraph("Original Text:", styles['Heading2'])
        story.append(original_heading)
        original_paragraph = Paragraph(original_text, styles['Normal'])
        story.append(original_paragraph)
        story.append(Spacer(1, 12))
        
        # Enhanced text section
        enhanced_heading = Paragraph("Enhanced Text:", styles['Heading2'])
        story.append(enhanced_heading)
        enhanced_paragraph = Paragraph(enhanced_text, styles['Normal'])
        story.append(enhanced_paragraph)
        
        # Build PDF document
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        # Encode as base64 for JSON transport
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        logging.info("PDF generated successfully")
        return pdf_base64
        
    except Exception as e:
        logging.error(f"PDF generation failed: {str(e)}")
        return ""


# ==============================================================================
# AGENT FUNCTION
# ==============================================================================

def consolidate_text_agent(state: AgentState) -> AgentState:
    """
    Consolidate text with synonym enhancements and generate comparison PDF.
    
    This agent applies the synonyms found by previous agents to create
    enhanced text, then generates a PDF document showing both original
    and enhanced versions side-by-side.
    
    Args:
        state: Current pipeline state with text and synonyms
        
    Returns:
        Updated state with enhanced_text and pdf_content
    """
    text = state.get("text", "")
    synonyms = state.get("synonyms", {})
    
    logging.info(f"Consolidating text with {len(synonyms)} synonym mappings...")
    
    # Apply synonym enhancements
    enhanced_text = _apply_synonym_enhancements(text, synonyms)
    state["enhanced_text"] = enhanced_text
    
    logging.info("Synonym enhancements applied")
    
    # Generate comparison PDF
    pdf_content = _generate_comparison_pdf(text, enhanced_text)
    state["pdf_content"] = pdf_content
    
    if pdf_content:
        logging.info(f"PDF generated ({len(pdf_content)} chars base64)")
    else:
        logging.warning("PDF generation returned empty content")
    
    logging.info("Text consolidation completed")
    return state
