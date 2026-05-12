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
        
        # Create question and context from Excel analysis
        question = f"What is the content and analysis of the Excel file '{filename}'?"
        answer = f"{title}\n\n{summary}"
        
        # Use extracted text as context
        contexts = [extracted_text] if extracted_text else ["No content extracted"]
        
        # Add file metadata to context
        if parsed_data:
            metadata_context = f"File Analysis: {parsed_data.get('file_analysis', {}).get('content_summary', '')}"
            contexts.append(metadata_context)
        
        logging.info(f"Preparing RAGAS dataset - question length: {len(question)}, answer length: {len(answer)}, context length: {len(contexts[0])}")
        
        # Create dataset for RAGAS
        # Note: context_recall requires 'ground_truth' field
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [extracted_text]  # Use extracted text as ground truth
        }
        
        dataset = Dataset.from_dict(data)
        logging.info(f"RAGAS dataset created with {len(dataset)} samples")
        
        # Perform RAGAS evaluation
        # Using faithfulness and answer_relevancy (context_recall requires ground_truth)
        logging.info("Starting RAGAS evaluation...")
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
            llm=azure_chat_model,
            embeddings=azure_embeddings,
        )
        
        logging.info(f"RAGAS evaluation completed. Result type: {type(result)}")
        logging.info(f"RAGAS result keys: {list(result.keys()) if hasattr(result, 'keys') else 'N/A'}")
        
        # Extract scores from result
        ragas_scores = _extract_ragas_scores(result)
        ragas_scores["evaluated"] = True
        ragas_scores["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        logging.info(f"Extracted RAGAS scores: {ragas_scores}")
        
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
    
    Args:
        result: RAGAS evaluation result
        
    Returns:
        Dictionary of extracted scores
    """
    scores = {}
    
    try:
        # RAGAS result can be a dict or have a to_pandas() method
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            for col in df.columns:
                if col in ['faithfulness', 'answer_relevancy', 'context_recall']:
                    scores[col] = float(df[col].iloc[0]) if len(df) > 0 else 0.0
        elif isinstance(result, dict):
            for key in ['faithfulness', 'answer_relevancy', 'context_recall']:
                if key in result:
                    value = result[key]
                    # Handle various result formats
                    if isinstance(value, (int, float)):
                        scores[key] = float(value)
                    elif isinstance(value, list) and len(value) > 0:
                        scores[key] = float(value[0])
                    else:
                        scores[key] = 0.0
        else:
            logging.warning(f"Unexpected RAGAS result type: {type(result)}")
            scores = {"error": "Unexpected result format"}
            
    except Exception as e:
        logging.error(f"Failed to extract RAGAS scores: {str(e)}")
        scores = {"error": f"Failed to extract scores: {str(e)}"}
    
    return scores
