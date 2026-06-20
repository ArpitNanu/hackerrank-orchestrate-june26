import os
import csv
from typing import List, Dict

from .schemas import ClaimObject, IssueType, RequirementResolution

# ---------------------------------------------------------
# CSV Loading Optimization
# ---------------------------------------------------------
# Cache the CSV rows so we only read the filesystem once per runtime.
_CACHED_REQUIREMENTS: List[Dict[str, str]] = []

def _load_requirements() -> List[Dict[str, str]]:
    global _CACHED_REQUIREMENTS
    if _CACHED_REQUIREMENTS:
        return _CACHED_REQUIREMENTS
        
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "dataset",
        "evidence_requirements.csv"
    )
    
    if not os.path.exists(csv_path):
        print(f"[Warning] Evidence requirements CSV not found at {csv_path}")
        return []
        
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _CACHED_REQUIREMENTS.append(row)
            
    return _CACHED_REQUIREMENTS

# ---------------------------------------------------------
# Matching Logic
# ---------------------------------------------------------

def _matches_applies_to(claimed_issue: IssueType, claimed_part: str, applies_to: str) -> bool:
    """
    Determines if the user's claimed issue or part matches the CSV's 'applies_to' column.
    """
    applies_to_lower = applies_to.lower()
    issue_str = claimed_issue.value.replace("_", " ").lower()
    part_str = claimed_part.replace("_", " ").lower()
    
    # 1. Direct substring match on issue (e.g., "dent" in "dent or scratch")
    if issue_str in applies_to_lower:
        return True
        
    # 2. Direct substring match on part (e.g., "screen" in "screen, keyboard, or trackpad")
    if part_str and part_str != "unknown" and part_str in applies_to_lower:
        return True
        
    # 3. Keyword overlap for complex multi-word enums like "broken_part" vs "broken"
    keywords = ["broken", "missing", "water", "stain", "torn", "crushed"]
    for kw in keywords:
        if kw in issue_str and kw in applies_to_lower:
            return True
            
    return False

# ---------------------------------------------------------
# Resolver
# ---------------------------------------------------------

def resolve_requirements(
    claim_object: ClaimObject, 
    claimed_issue: IssueType,
    claimed_part: str = ""
) -> RequirementResolution:
    """
    Loads and resolves the minimum image evidence requirement.
    This module performs lookup only. No LLM calls. No image analysis.
    
    Args:
        claim_object: The overarching object type (e.g., CAR).
        claimed_issue: The specific issue (e.g., DENT).
        claimed_part: The specific part (optional, needed because laptop CSV rules target parts).
        
    Returns:
        RequirementResolution: Typed schema containing the requirement ID and text.
    """
    rows = _load_requirements()
    
    # Priority 1: Match claim_object AND applies_to specifically
    for row in rows:
        if row["claim_object"].lower() == claim_object.value.lower():
            if _matches_applies_to(claimed_issue, claimed_part, row["applies_to"]):
                return RequirementResolution(
                    requirement_id=row["requirement_id"],
                    requirement_text=row["minimum_image_evidence"]
                )
                
    # Priority 2: Fall back to object-specific generic rules
    # (Matches the object but the rule applies generally)
    for row in rows:
        if row["claim_object"].lower() == claim_object.value.lower():
            if "general" in row["applies_to"].lower() or "all" in row["applies_to"].lower():
                return RequirementResolution(
                    requirement_id=row["requirement_id"],
                    requirement_text=row["minimum_image_evidence"]
                )
                
    # Priority 3: Fall back to generic 'all' rules
    for row in rows:
        if row["claim_object"].lower() == "all" and "general claim review" in row["applies_to"].lower():
            return RequirementResolution(
                requirement_id=row["requirement_id"],
                requirement_text=row["minimum_image_evidence"]
            )
            
    # Final safety fallback (if CSV is missing or completely unmatched)
    return RequirementResolution(
        requirement_id="REQ_DEFAULT_GENERAL",
        requirement_text="The claimed object and relevant part should be visible clearly enough to inspect the claimed condition."
    )
