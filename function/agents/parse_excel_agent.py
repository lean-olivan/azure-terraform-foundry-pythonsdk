import logging
import pandas as pd
import openpyxl
from io import BytesIO
from typing import TypedDict, Dict, Any, List

class ExcelAgentState(TypedDict):
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

def parse_excel_agent(state: ExcelAgentState) -> ExcelAgentState:
    """Parse and analyze Excel file content"""
    excel_content = state.get("excel_content", b"")
    filename = state.get("filename", "unknown.xlsx")
    
    if not excel_content:
        logging.error("No Excel content provided to parse_excel_agent")
        state["parsed_data"] = {"error": "No Excel content provided"}
        return state
    
    try:
        # Parse Excel file
        excel_file = BytesIO(excel_content)
        
        # Get sheet names using openpyxl
        workbook = openpyxl.load_workbook(excel_file, read_only=True)
        sheet_names = workbook.sheetnames
        workbook.close()
        
        # Parse each sheet with pandas
        all_sheets_data = {}
        all_text_content = []
        total_rows = 0
        total_columns = 0
        total_cells = 0
        
        for sheet_name in sheet_names:
            try:
                # Read sheet with pandas
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
                
                # Get basic info
                sheet_rows, sheet_columns = df.shape
                sheet_cells = df.notna().sum().sum()
                
                total_rows += sheet_rows
                total_columns += sheet_columns
                total_cells += sheet_cells
                
                # Convert to structured data
                sheet_data = {
                    "sheet_name": sheet_name,
                    "rows": sheet_rows,
                    "columns": sheet_columns,
                    "non_empty_cells": int(sheet_cells),
                    "data": df.fillna("").astype(str).values.tolist(),
                    "has_headers": _detect_headers(df),
                    "data_types": _detect_data_types(df),
                }
                
                all_sheets_data[sheet_name] = sheet_data
                
                # Extract text content for analysis
                text_content = _extract_text_from_sheet(df, sheet_name)
                all_text_content.append(text_content)
                
                logging.info(f"Parsed sheet '{sheet_name}': {sheet_rows}x{sheet_columns}, {sheet_cells} non-empty cells")
                
            except Exception as e:
                logging.error(f"Error parsing sheet '{sheet_name}': {str(e)}")
                all_sheets_data[sheet_name] = {
                    "sheet_name": sheet_name,
                    "error": str(e),
                    "rows": 0,
                    "columns": 0,
                    "non_empty_cells": 0,
                    "data": []
                }
        
        # Combine all text content
        combined_text = "\n\n".join(all_text_content)
        
        # Create comprehensive parsed data
        parsed_data = {
            "file_type": "excel",
            "filename": filename,
            "total_sheets": len(sheet_names),
            "sheet_names": sheet_names,
            "total_rows": total_rows,
            "total_columns": total_columns,
            "total_cells": total_cells,
            "sheets": all_sheets_data,
            "extracted_text": combined_text,
            "text_length": len(combined_text),
            "file_analysis": {
                "complexity": _assess_complexity(all_sheets_data),
                "data_categories": _identify_data_categories(all_sheets_data),
                "structure_type": _identify_structure_type(all_sheets_data),
                "content_summary": _generate_content_summary(all_sheets_data)
            }
        }
        
        state["parsed_data"] = parsed_data
        state["text"] = combined_text  # Set text for downstream agents
        
        logging.info(f"Successfully parsed Excel file '{filename}': {len(sheet_names)} sheets, {total_rows} rows, {total_cells} cells")
        
    except Exception as e:
        logging.error(f"Error parsing Excel file '{filename}': {str(e)}")
        state["parsed_data"] = {
            "error": str(e),
            "file_type": "excel",
            "filename": filename,
            "total_sheets": 0,
            "sheet_names": [],
            "total_rows": 0,
            "total_columns": 0,
            "total_cells": 0,
            "sheets": {},
            "extracted_text": ""
        }
        state["text"] = ""
    
    return state

def _detect_headers(df: pd.DataFrame) -> bool:
    """Detect if the first row likely contains headers"""
    if df.shape[0] < 2:
        return False
    
    first_row = df.iloc[0].astype(str)
    other_rows = df.iloc[1:].astype(str)
    
    # Check if first row has different data types than other rows
    first_row_types = set(first_row.apply(type))
    other_row_values = other_rows.values.flatten() if len(other_rows) > 0 else []
    other_row_types = set(type(val) for val in other_row_values) if len(other_row_values) > 0 else set()
    
    # If first row has mostly strings and other rows have mixed types, likely headers
    if (str in first_row_types and 
        len(first_row_types) == 1 and 
        len(other_row_types) > 1):
        return True
    
    # Check for common header patterns
    header_indicators = ['date', 'name', 'id', 'amount', 'price', 'status', 'type', 'category']
    first_row_lower = first_row.str.lower()
    
    for cell in first_row_lower:
        if any(indicator in str(cell) for indicator in header_indicators):
            return True
    
    return False

def _detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect data types in each column"""
    data_types = {}
    
    for col_idx in range(df.shape[1]):
        column = df.iloc[:, col_idx].dropna()
        
        if len(column) == 0:
            data_types[f"column_{col_idx}"] = "empty"
            continue
        
        # Sample first few non-empty values
        sample_values = column.head(5).astype(str).tolist()
        
        # Detect data type patterns
        if all(_is_date(val) for val in sample_values if val.strip()):
            data_types[f"column_{col_idx}"] = "date"
        elif all(_is_numeric(val) for val in sample_values if val.strip()):
            data_types[f"column_{col_idx}"] = "numeric"
        elif all(_is_currency(val) for val in sample_values if val.strip()):
            data_types[f"column_{col_idx}"] = "currency"
        elif all(_is_percentage(val) for val in sample_values if val.strip()):
            data_types[f"column_{col_idx}"] = "percentage"
        else:
            data_types[f"column_{col_idx}"] = "text"
    
    return data_types

def _is_date(value: str) -> bool:
    """Check if the value looks like a date"""
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
    ]
    
    import re
    return any(re.match(pattern, value.strip()) for pattern in date_patterns)

def _is_numeric(value: str) -> bool:
    """Check if value is numeric"""
    try:
        float(value.replace(',', '').strip())
        return True
    except ValueError:
        return False

def _is_currency(value: str) -> bool:
    """Check if value looks like currency"""
    return (value.strip().startswith(('$', '€', '£', '¥')) or 
            value.strip().endswith(('USD', 'EUR', 'GBP', 'JPY')))

def _is_percentage(value: str) -> bool:
    """Check if value looks like a percentage"""
    return value.strip().endswith('%')

def _extract_text_from_sheet(df: pd.DataFrame, sheet_name: str) -> str:
    """Extract readable text from a sheet"""
    text_content = []
    text_content.append(f"--- Sheet: {sheet_name} ---")
    
    for idx, row in df.iterrows():
        # Convert row to text, joining cells with separators
        row_text = " | ".join([str(cell) if pd.notna(cell) else "" for cell in row])
        if row_text.strip():
            text_content.append(row_text)
    
    return "\n".join(text_content)

def _assess_complexity(sheets_data: Dict[str, Any]) -> str:
    """Assess the complexity of the Excel file"""
    total_sheets = len(sheets_data)
    total_cells = sum(sheet.get("non_empty_cells", 0) for sheet in sheets_data.values())
    
    if total_sheets <= 2 and total_cells <= 100:
        return "Simple"
    elif total_sheets <= 5 and total_cells <= 500:
        return "Medium"
    else:
        return "Complex"

def _identify_data_categories(sheets_data: Dict[str, Any]) -> List[str]:
    """Identify potential data categories based on content"""
    categories = []
    
    # Look for common patterns in sheet names and data
    sheet_names = [sheet.get("sheet_name", "").lower() for sheet in sheets_data.values()]
    
    if any(keyword in " ".join(sheet_names) for keyword in ["financial", "budget", "revenue", "cost"]):
        categories.append("Financial")
    
    if any(keyword in " ".join(sheet_names) for keyword in ["risk", "assessment", "mitigation"]):
        categories.append("Risk Management")
    
    if any(keyword in " ".join(sheet_names) for keyword in ["project", "task", "timeline"]):
        categories.append("Project Management")
    
    if any(keyword in " ".join(sheet_names) for keyword in ["customer", "client", "sales"]):
        categories.append("Customer Data")
    
    if any(keyword in " ".join(sheet_names) for keyword in ["inventory", "stock", "product"]):
        categories.append("Inventory")
    
    if not categories:
        categories.append("General")
    
    return categories

def _identify_structure_type(sheets_data: Dict[str, Any]) -> str:
    """Identify the structure type of the Excel file"""
    sheet_count = len(sheets_data)
    
    if sheet_count == 1:
        return "Single Sheet"
    elif sheet_count <= 3:
        return "Multi-Sheet"
    else:
        return "Complex Workbook"

def _generate_content_summary(sheets_data: Dict[str, Any]) -> str:
    """Generate a brief summary of the content"""
    sheet_names = [sheet.get("sheet_name", "Unknown") for sheet in sheets_data.values()]
    total_cells = sum(sheet.get("non_empty_cells", 0) for sheet in sheets_data.values())
    
    return f"Excel workbook with {len(sheet_names)} sheets ({', '.join(sheet_names)}) containing {total_cells} data cells"