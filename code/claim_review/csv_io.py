import os
import csv
from typing import List, Dict, Any
from .schemas import FinalOutputRow

def _get_dataset_path(filename: str) -> str:
    """Helper to resolve the absolute path to a dataset file."""
    # Assuming code layout:
    # hackerrank-orchestrate-june26/
    # ├── code/
    # │   └── claim_review/
    # │       └── csv_io.py
    # └── dataset/
    #     └── claims.csv
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "dataset", filename)

def read_claims(path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Reads the input claims from dataset/claims.csv.
    Returns a list of dictionaries mapping column names to values.
    """
    claims = []
    path = path or _get_dataset_path("claims.csv")
    if not os.path.exists(path):
        print(f"[Error] claims.csv not found at {path}")
        return claims
        
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claims.append(row)
            
    return claims

def read_user_history(path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Reads dataset/user_history.csv.
    Returns a dictionary mapping user_id to their complete history row.
    """
    history_map = {}
    path = path or _get_dataset_path("user_history.csv")
    if not os.path.exists(path):
        print(f"[Warning] user_history.csv not found at {path}")
        return history_map
        
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row.get("user_id")
            if user_id:
                history_map[user_id] = row
                
    return history_map

def read_requirements(path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Reads dataset/evidence_requirements.csv.
    (Also cached and used by requirements_resolver.py).
    """
    requirements = []
    path = path or _get_dataset_path("evidence_requirements.csv")
    if not os.path.exists(path):
        print(f"[Warning] evidence_requirements.csv not found at {path}")
        return requirements
        
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            requirements.append(row)
            
    return requirements

def write_output(output_path: str, rows: List[FinalOutputRow]) -> None:
    """
    Writes the final pipeline outputs to the specified output CSV file.
    
    Requirements:
    - Output columns must exactly match the challenge specification (from FinalOutputRow schema).
    - Enums are serialized as their underlying string values.
    - Lists are written as semicolon-separated strings (handled prior to this step 
      by the FinalOutputRow schema mapping, but explicitly guarded here).
    """
    if not rows:
        print("[Warning] No rows to write to output CSV.")
        return
        
    # Extract headers sequentially from the Pydantic model's fields
    field_names = list(FinalOutputRow.model_fields.keys())
    
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        
        for row in rows:
            # model_dump converts Pydantic structures to dictionaries.
            # In Pydantic v2, string enums are natively converted to their string values.
            row_dict = row.model_dump()
            
            # Double check enum stringification just in case (e.g. ClaimStatus.SUPPORTED -> "supported")
            if hasattr(row.issue_type, "value"):
                row_dict["issue_type"] = row.issue_type.value
            if hasattr(row.claim_status, "value"):
                row_dict["claim_status"] = row.claim_status.value
            if hasattr(row.severity, "value"):
                row_dict["severity"] = row.severity.value
                
            # List stringification (e.g. ["img_1", "img_2"] -> "img_1;img_2").
            # The FinalOutputRow already typed risk_flags and supporting_image_ids as strings.
            # If any raw lists somehow bypassed this and made it here, handle them dynamically.
            for key, val in row_dict.items():
                if isinstance(val, list):
                    row_dict[key] = ";".join(str(v) for v in val)
            
            writer.writerow(row_dict)
