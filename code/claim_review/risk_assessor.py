from typing import List, Dict, Any, Optional
from .schemas import ImageQualification, InspectionObservation, RiskAssessmentResult, RiskFlag, UserHistoryRow

# ---------------------------------------------------------
# Configurable Constants
# ---------------------------------------------------------
MAX_REJECTED_CLAIMS = 0
MAX_RECENT_CLAIMS = 2

# ---------------------------------------------------------
# Risk Assessment Logic
# ---------------------------------------------------------

def assess_risk(
    qualification: ImageQualification,
    observations: List[InspectionObservation],
    user_history: UserHistoryRow
) -> RiskAssessmentResult:
    """
    Aggregates risk flags from images, observations, and user history.
    
    IMPORTANT ARCHITECTURAL RULE:
    Risk assessment MUST NEVER modify the deterministic evidence validation output
    (supported, contradicted, not_enough_information). 
    
    Why? 
    Historical risk is context, not visual evidence. A high-risk user can 
    still submit a 100% valid claim, and a low-risk user can still submit a fraudulent one.
    If we allow risk scores to override visual evidence, we destroy the auditability and
    fairness of the system. 
    
    The Evidence Validation stage strictly rules on what is visible in the image.
    This Risk Assessment stage flags the transaction for potential downstream human 
    review without altering the underlying evidentiary truth.
    """
    # Use a set to automatically handle Rule 4 (Remove duplicate flags)
    flags = set()
    
    # 1. Copy image quality flags from ImageQualification.
    for flag in qualification.quality_flags:
        flags.add(flag)
        
    # 2. Add user_history_risk based on thresholds.
    if user_history.rejected_claim > MAX_REJECTED_CLAIMS or user_history.last_90_days_claim_count > MAX_RECENT_CLAIMS:
        flags.add(RiskFlag.USER_HISTORY_RISK)
        
    # Example logic using the other new history fields:
    # If the user has a "fraud_warning" in their history flags, automatically flag.
    if "fraud_warning" in user_history.history_flags.lower():
        flags.add(RiskFlag.USER_HISTORY_RISK)
        
    # 3. Add manual_review_required when specific qualification criteria fail.
    if not qualification.image_usable or not qualification.object_correct or not qualification.claim_part_visible:
        flags.add(RiskFlag.MANUAL_REVIEW_REQUIRED)
        
    # 4. Cleanup: Remove NONE if other risks exist.
    if RiskFlag.NONE in flags and len(flags) > 1:
        flags.remove(RiskFlag.NONE)
        
    # 5. If no risks exist, ensure RiskFlag.NONE is the only element.
    if not flags:
        flags.add(RiskFlag.NONE)
        
    # Convert back to list for Pydantic schema validation.
    return RiskAssessmentResult(risk_flags=list(flags))
