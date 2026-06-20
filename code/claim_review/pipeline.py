from typing import List, Dict, Any, Optional

from .schemas import (
    ClaimObject, ClaimExtraction, ImageQualification, InspectionObservation,
    EvidenceValidationResult, RiskAssessmentResult, FinalOutputRow,
    IssueType, ClaimStatus, Severity, RiskFlag,
    CarPart, LaptopPart, PackagePart, UserHistoryRow
)
from .claim_extractor import extract_claim
from .image_qualifier import qualify_images
from .decision_rules import evaluate_claim
from .inspectors import CarInspector, LaptopInspector, PackageInspector
from .risk_assessor import assess_risk
from .requirements_resolver import resolve_requirements

# ---------------------------------------------------------
# Inspector Selection
# ---------------------------------------------------------
# Maps claim_object to the correct domain-specific inspector.
# This is the only routing logic in the pipeline.

INSPECTOR_MAP = {
    ClaimObject.CAR: CarInspector,
    ClaimObject.LAPTOP: LaptopInspector,
    ClaimObject.PACKAGE: PackageInspector,
}


def _get_unknown_part(claim_object: ClaimObject):
    """Returns the correct UNKNOWN part enum for a given claim object type."""
    if claim_object == ClaimObject.CAR:
        return CarPart.UNKNOWN
    elif claim_object == ClaimObject.LAPTOP:
        return LaptopPart.UNKNOWN
    elif claim_object == ClaimObject.PACKAGE:
        return PackagePart.UNKNOWN
    return CarPart.UNKNOWN


def _aggregate_severity(observations: List[InspectionObservation]) -> Severity:
    """
    Derives a single severity from multiple observations.
    Takes the highest severity across all images where damage is visible.
    """
    severity_order = {
        Severity.NONE: 0,
        Severity.UNKNOWN: 1,
        Severity.LOW: 2,
        Severity.MEDIUM: 3,
        Severity.HIGH: 4,
    }
    max_severity = Severity.NONE
    for obs in observations:
        if obs.damage_visible and severity_order.get(obs.severity, 0) > severity_order.get(max_severity, 0):
            max_severity = obs.severity
    return max_severity



# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------
# Execution order:
#   1. Claim Extraction       (LLM — text only)
#   2. Image Qualification    (LLM — vision)
#   3. Object Inspection      (LLM — vision, domain-specific)
#   4. Evidence Validation    (deterministic code)
#   5. Risk Assessment        (deterministic code)
#   6. Final Output Assembly  (deterministic code)
#
# Each stage receives typed schemas from the previous stage.
# No business logic lives in this file — only orchestration.

def process_claim(
    user_id: str,
    user_claim: str,
    claim_object: str,
    image_paths: List[str],
    user_history: Optional[UserHistoryRow] = None,
    evidence_requirement: Optional[str] = None,
    api_key: Optional[str] = None
) -> FinalOutputRow:
    """
    Orchestrates the full claim review pipeline.
    
    Args:
        user_id: Unique user identifier.
        user_claim: Raw text of the user's damage claim.
        claim_object: Object type string (car/laptop/package).
        image_paths: List of paths to submitted evidence images.
        evidence_requirement: Optional minimum evidence standard.
        api_key: Optional OpenAI API key.
        
    Returns:
        FinalOutputRow: Complete row matching the challenge CSV schema.
    """
    claim_object_enum = ClaimObject(claim_object)
    image_paths_str = ";".join(image_paths)

    # --------------------------------------------------
    # Stage 1: Claim Extraction
    # --------------------------------------------------
    # Extract what the user is alleging. No truth determination.
    try:
        claim = extract_claim(user_claim, claim_object_enum, api_key)
    except Exception as e:
        print(f"[Pipeline Error] Stage 1 (Claim Extraction) failed: {e}")
        claim = ClaimExtraction(
            claim_object=claim_object_enum,
            claimed_issue=IssueType.UNKNOWN,
            claimed_part=_get_unknown_part(claim_object_enum)
        )

    # --------------------------------------------------
    # Stage 1.5: Resolve Evidence Requirements
    # --------------------------------------------------
    # Look up the minimum evidence standard from evidence_requirements.csv
    # based on what the user is claiming. This feeds into image qualification
    # so the LLM knows what evidence bar to observe against.
    try:
        requirement = resolve_requirements(
            claim_object=claim_object_enum,
            claimed_issue=claim.claimed_issue,
            claimed_part=claim.claimed_part.value
        )
        resolved_requirement = requirement.requirement_text
    except Exception as e:
        print(f"[Pipeline Error] Stage 1.5 (Requirements Resolution) failed: {e}")
        resolved_requirement = evidence_requirement  # Fall back to caller-provided value

    # --------------------------------------------------
    # Stage 2: Image Qualification
    # --------------------------------------------------
    # Determine whether the images are suitable for evaluation.
    # The resolved evidence requirement text is passed so the LLM
    # can observe against the specific minimum evidence standard.
    try:
        qualification = qualify_images(image_paths, claim, resolved_requirement, api_key)
    except Exception as e:
        print(f"[Pipeline Error] Stage 2 (Image Qualification) failed: {e}")
        qualification = ImageQualification(
            valid_image=False,
            image_usable=False,
            object_correct=False,
            part_visible=False,
            claim_part_visible=False,
            quality_flags=[RiskFlag.MANUAL_REVIEW_REQUIRED]
        )

    # --------------------------------------------------
    # Stage 3: Object Inspection
    # --------------------------------------------------
    # Inspect images for visual observations. Skipped if images are unusable.
    observations: List[InspectionObservation] = []

    if qualification.image_usable:
        try:
            inspector_class = INSPECTOR_MAP.get(claim_object_enum)
            if inspector_class:
                inspector = inspector_class(api_key)
                observations = inspector.inspect(image_paths)
        except Exception as e:
            print(f"[Pipeline Error] Stage 3 (Object Inspection) failed: {e}")

    # --------------------------------------------------
    # Stage 4: Deterministic Evidence Validation
    # --------------------------------------------------
    # Compare extraction against observations using coded rules.
    try:
        validation = evaluate_claim(claim, observations, qualification)
    except Exception as e:
        print(f"[Pipeline Error] Stage 4 (Evidence Validation) failed: {e}")
        validation = EvidenceValidationResult(
            evidence_standard_met=False,
            evidence_standard_met_reason="Pipeline error during evidence validation.",
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
            claim_status_justification="An error occurred during evidence validation.",
            supporting_image_ids=["none"]
        )

    # --------------------------------------------------
    # Stage 5: Risk Assessment
    # --------------------------------------------------
    # Aggregate risk flags from all prior stages + user history.
    # IMPORTANT: Risk assessment NEVER modifies supported/contradicted/not_enough_information.
    # It only adds contextual risk_flags.
    try:
        risk = assess_risk(qualification, observations, user_history or {})
    except Exception as e:
        print(f"[Pipeline Error] Stage 5 (Risk Assessment) failed: {e}")
        risk = RiskAssessmentResult(risk_flags=[RiskFlag.MANUAL_REVIEW_REQUIRED])

    # --------------------------------------------------
    # Stage 6: Final Output Assembly
    # --------------------------------------------------
    # Assemble all results into the challenge CSV row format.
    severity = _aggregate_severity(observations) if observations else Severity.UNKNOWN

    # --------------------------------------------------
    # Determine issue_type and object_part for output row.
    # --------------------------------------------------
    # Selection hierarchy (deterministic, per claim_status):
    #
    # SUPPORTED:
    #   Use the matching observation (issue + part confirmed visually).
    #   Fallback to claimed values if no exact match found.
    #
    # CONTRADICTED:
    #   Use the BEST visual observation — what was actually seen.
    #   This may differ from the claim (that's the contradiction).
    #   If no observations exist (e.g., wrong object), use UNKNOWN/NONE.
    #   Examples from sample data:
    #     user_005: claimed dent → observed scratch → output scratch
    #     user_020: claimed damage → observed none → output none
    #     user_033: wrong object → output unknown
    #
    # NOT_ENOUGH_INFORMATION:
    #   Use the best visual observation if one exists.
    #   If no observations exist, use UNKNOWN.
    #   Examples from sample data:
    #     user_002: observed broken_part → output broken_part
    #     user_006: nothing clear → output unknown
    #     user_032: no usable images → output unknown

    from .decision_rules import MIN_CONFIDENCE

    if validation.claim_status == ClaimStatus.SUPPORTED and observations:
        # SUPPORTED: prefer the observation that matched the claim
        matched_obs = next(
            (obs for obs in observations
             if obs.confidence >= MIN_CONFIDENCE
             and obs.visible_issue == claim.claimed_issue
             and obs.visible_part == claim.claimed_part),
            None
        )
        if matched_obs:
            issue_type = matched_obs.visible_issue
            object_part = matched_obs.visible_part.value
        else:
            # Fallback: claim was supported but no exact match in filtered obs
            issue_type = claim.claimed_issue
            object_part = claim.claimed_part.value

    elif validation.claim_status == ClaimStatus.CONTRADICTED and observations:
        # CONTRADICTED: report what was ACTUALLY observed, not what was claimed.
        # Pick the highest-confidence observation on the claimed part.
        best_obs = None
        best_confidence = -1.0
        for obs in observations:
            if obs.confidence >= MIN_CONFIDENCE and obs.confidence > best_confidence:
                best_obs = obs
                best_confidence = obs.confidence

        if best_obs:
            issue_type = best_obs.visible_issue
            object_part = best_obs.visible_part.value
        else:
            # No confident observations — contradiction from qualification checks
            issue_type = IssueType.UNKNOWN
            object_part = claim.claimed_part.value

    elif validation.claim_status == ClaimStatus.NOT_ENOUGH_INFORMATION and observations:
        # NOT_ENOUGH_INFO: report the best observation if available.
        best_obs = None
        best_confidence = -1.0
        for obs in observations:
            if obs.confidence >= MIN_CONFIDENCE and obs.confidence > best_confidence:
                best_obs = obs
                best_confidence = obs.confidence

        if best_obs:
            issue_type = best_obs.visible_issue
            object_part = best_obs.visible_part.value
        else:
            issue_type = IssueType.UNKNOWN
            object_part = claim.claimed_part.value

    else:
        # No observations at all (images unusable, wrong object, etc.)
        # Check qualification to decide between UNKNOWN and claimed values.
        if not qualification.object_correct:
            issue_type = IssueType.UNKNOWN
            object_part = "unknown"
        elif not qualification.claim_part_visible:
            issue_type = IssueType.UNKNOWN
            object_part = claim.claimed_part.value
        else:
            issue_type = claim.claimed_issue
            object_part = claim.claimed_part.value

    return FinalOutputRow(
        user_id=user_id,
        image_paths=image_paths_str,
        user_claim=user_claim,
        claim_object=claim_object_enum.value,
        evidence_standard_met=validation.evidence_standard_met,
        evidence_standard_met_reason=validation.evidence_standard_met_reason,
        risk_flags=";".join(flag.value for flag in risk.risk_flags),
        issue_type=issue_type,
        object_part=object_part,
        claim_status=validation.claim_status,
        claim_status_justification=validation.claim_status_justification,
        supporting_image_ids=";".join(validation.supporting_image_ids),
        valid_image=qualification.valid_image,
        severity=severity
    )
