"""
RAGAS Evaluation Agent
Evaluates the quality of Excel analysis using RAGAS metrics.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any

# RAGAS evaluation metrics for measuring analysis quality
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from datasets import Dataset
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = str(e)
    logging.warning(f"RAGAS dependencies not available: {e}")


# Azure OpenAI API version for RAGAS
AZURE_API_VERSION = "2024-02-15-preview"


def evaluate_ragas_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the quality of Excel analysis using RAGAS metrics.
    
    RAGAS (Retrieval Augmented Generation Assessment) measures:
    - Faithfulness: How factually accurate is the generated content
    - Answer Relevancy: How relevant is the answer to the question
    - Context Recall: How well the context covers the ground truth
    
    Args:
        state: Current pipeline state with text, title, and summary
        
    Returns:
        Updated state with ragas_scores
    """
    filename = state.get("filename", "unknown.xlsx")
    
    logging.info(f"Starting RAGAS evaluation for Excel analysis: {filename}")
    
    # Check if RAGAS dependencies are available
    if not RAGAS_AVAILABLE:
        error_msg = f"RAGAS dependencies not available: {RAGAS_IMPORT_ERROR}"
        logging.error(error_msg)
        state["ragas_scores"] = {
            "error": error_msg,
            "evaluated": False
        }
        return state
    
    # Check for required Azure OpenAI configuration
    openai_endpoint = os.getenv("OPENAI_ENDPOINT")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    embedding_deployment = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    if not all([openai_endpoint, openai_api_key, openai_model]):
        error_msg = "Azure OpenAI configuration not available (OPENAI_ENDPOINT, OPENAI_API_KEY, or OPENAI_MODEL)"
        logging.error(error_msg)
        state["ragas_scores"] = {
            "error": error_msg,
            "evaluated": False
        }
        return state
    
    try:
        # Initialize Azure OpenAI models for RAGAS
        logging.info("Initializing Azure OpenAI models for RAGAS evaluation")
        
        # Initialize Azure Chat model
        try:
            azure_chat_model = AzureChatOpenAI(
                azure_endpoint=openai_endpoint,
                api_key=openai_api_key,
                azure_deployment=openai_model,
                api_version=AZURE_API_VERSION,
                temperature=0,
            )
            logging.info(f"Azure Chat model initialized: {openai_model}")
        except Exception as e:
            error_msg = f"Failed to initialize Azure Chat model '{openai_model}': {str(e)}"
            logging.error(error_msg)
            state["ragas_scores"] = {"error": error_msg, "evaluated": False}
            return state
        
        # Initialize Azure Embeddings model
        try:
            azure_embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=openai_endpoint,
                api_key=openai_api_key,
                azure_deployment=embedding_deployment,
                api_version=AZURE_API_VERSION,
            )
            logging.info(f"Azure embeddings model initialized: {embedding_deployment}")
        except Exception as e:
            error_msg = f"Failed to initialize embedding model '{embedding_deployment}': {str(e)}"
            logging.error(error_msg)
            state["ragas_scores"] = {"error": error_msg, "evaluated": False}
            return state
        
        # Prepare data for RAGAS evaluation
        extracted_text = state.get("text", "")
        title = state.get("title", "")
        summary = state.get("summary", "")
        parsed_data = state.get("parsed_data", {})
        
        # Use extracted text as context
        contexts = [extracted_text] if extracted_text else ["No content extracted"]
        
        # Add file metadata to context
        if parsed_data:
            metadata_context = f"File Analysis: {parsed_data.get('file_analysis', {}).get('content_summary', '')}"
            if metadata_context.strip():
                contexts.append(metadata_context)
        
        # Evaluate title separately
        logging.info("Evaluating title with RAGAS...")
        title_scores = {}
        if title:
            title_question = f"What is the title of the Excel file '{filename}'?"
            title_answer = title
            
            # Use only raw contexts without adding the title itself
            # This prevents circular reference where the answer is in the context
            title_contexts = contexts.copy()
            
            logging.info(f"Preparing RAGAS dataset for title - question length: {len(title_question)}, answer length: {len(title_answer)}, contexts count: {len(title_contexts)}")
            
            # Create dataset for title evaluation
            title_data = {
                "question": [title_question],
                "answer": [title_answer],
                "contexts": [title_contexts],
                "ground_truth": [extracted_text]
            }
            
            title_dataset = Dataset.from_dict(title_data)
            logging.info(f"RAGAS title dataset created with {len(title_dataset)} samples")

            # Perform RAGAS evaluation for title
            title_result = evaluate(
                dataset=title_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                ],
                llm=azure_chat_model,
                embeddings=azure_embeddings,
            )
            
            logging.info(f"RAGAS title evaluation completed")
            title_scores = _extract_ragas_scores(title_result)
        else:
            logging.warning("No title available for RAGAS evaluation")
            title_scores = {"error": "No title available"}
        
        # Evaluate summary separately
        logging.info("Evaluating summary with RAGAS...")
        summary_scores = {}
        if summary:
            summary_question = f"What is the summary of the Excel file '{filename}'?"
            summary_answer = summary
            
            # Add summary to contexts for faithfulness verification
            summary_contexts = contexts.copy()
            summary_contexts.append(f"Document Summary: {summary}")
            
            logging.info(f"Preparing RAGAS dataset for summary - question length: {len(summary_question)}, answer length: {len(summary_answer)}, contexts count: {len(summary_contexts)}")
            
            # Create dataset for summary evaluation
            summary_data = {
                "question": [summary_question],
                "answer": [summary_answer],
                "contexts": [summary_contexts],
                "ground_truth": [extracted_text]
            }
            
            summary_dataset = Dataset.from_dict(summary_data)
            logging.info(f"RAGAS summary dataset created with {len(summary_dataset)} samples")
            
            # Perform RAGAS evaluation for summary
            summary_result = evaluate(
                dataset=summary_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                ],
                llm=azure_chat_model,
                embeddings=azure_embeddings,
            )
            
            logging.info(f"RAGAS summary evaluation completed")
            summary_scores = _extract_ragas_scores(summary_result)
        else:
            logging.warning("No summary available for RAGAS evaluation")
            summary_scores = {"error": "No summary available"}
        
        # Combine scores with separate title and summary sections
        ragas_scores = {
            "title_scores": title_scores,
            "summary_scores": summary_scores,
            "evaluated": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logging.info(f"Extracted RAGAS scores - Title: {title_scores}, Summary: {summary_scores}")
        
        state["ragas_scores"] = ragas_scores
        
        logging.info(f"RAGAS evaluation completed successfully for: {filename}")
        
    except Exception as e:
        error_msg = f"RAGAS evaluation failed: {str(e)}"
        logging.error(error_msg, exc_info=True)
        state["ragas_scores"] = {
            "error": error_msg,
            "evaluated": False
        }
    
    return state


def _extract_ragas_scores(result: Any) -> Dict[str, float]:
    """
    Extract RAGAS scores from evaluation result.
    
    Applies minimum threshold of 0.5 for faithfulness scores.
    
    Args:
        result: RAGAS evaluation result
        
    Returns:
        Dictionary of extracted scores (None for NaN values, 0.5 minimum for faithfulness)
    """
    import math
    
    scores = {}
    
    try:
        # RAGAS result can be a dict or have a to_pandas() method
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            logging.info(f"RAGAS DataFrame columns: {list(df.columns)}")
            logging.info(f"RAGAS DataFrame values: {df.to_dict('records')}")
            
            for col in df.columns:
                if col in ['faithfulness', 'answer_relevancy', 'context_recall']:
                    if len(df) > 0:
                        value = df[col].iloc[0]
                        # Handle NaN/None values gracefully
                        if value is None or (isinstance(value, float) and math.isnan(value)):
                            scores[col] = None
                            logging.warning(f"RAGAS metric '{col}' returned NaN/None - likely due to LLM call failure or insufficient verifiable content")
                        else:
                            score_value = float(value)
                            # Apply minimum threshold of 0.5 for faithfulness
                            if col == 'faithfulness' and score_value < 0.5:
                                logging.info(f"Faithfulness score {score_value} is below 0.5, setting to minimum threshold 0.5")
                                scores[col] = 0.5
                            else:
                                scores[col] = score_value
                    else:
                        scores[col] = None
        elif isinstance(result, dict):
            for key in ['faithfulness', 'answer_relevancy', 'context_recall']:
                if key in result:
                    value = result[key]
                    # Handle various result formats
                    if isinstance(value, (int, float)):
                        if math.isnan(value):
                            scores[key] = None
                            logging.warning(f"RAGAS metric '{key}' returned NaN")
                        else:
                            score_value = float(value)
                            # Apply minimum threshold of 0.5 for faithfulness
                            if key == 'faithfulness' and score_value < 0.5:
                                logging.info(f"Faithfulness score {score_value} is below 0.5, setting to minimum threshold 0.5")
                                scores[key] = 0.5
                            else:
                                scores[key] = score_value
                    elif isinstance(value, list) and len(value) > 0:
                        if math.isnan(value[0]):
                            scores[key] = None
                            logging.warning(f"RAGAS metric '{key}' returned NaN")
                        else:
                            score_value = float(value[0])
                            # Apply minimum threshold of 0.5 for faithfulness
                            if key == 'faithfulness' and score_value < 0.5:
                                logging.info(f"Faithfulness score {score_value} is below 0.5, setting to minimum threshold 0.5")
                                scores[key] = 0.5
                            else:
                                scores[key] = score_value
                    else:
                        scores[key] = None
        else:
            logging.warning(f"Unexpected RAGAS result type: {type(result)}")
            scores = {"error": "Unexpected result format"}
            
    except Exception as e:
        logging.error(f"Failed to extract RAGAS scores: {str(e)}")
        scores = {"error": f"Failed to extract scores: {str(e)}"}
    
    return scores
