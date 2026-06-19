from typing import List, Optional

from .schemas import (
    ClaimObject, ClaimExtraction, ImageQualification, InspectionObservation,
    EvidenceValidationResult, RiskAssessmentResult, FinalOutputRow,
    IssueType, ClaimStatus, Severity, RiskFlag,
    CarPart, LaptopPart, PackagePart
)
from .claim_extractor import extract_claim
from .image_qualifier import qualify_images
from .decision_rules import evaluate_claim
from .inspectors import CarInspector, LaptopInspector, PackageInspector

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


def _aggregate_risk_flags(
    qualification: ImageQualification,
    observations: List[InspectionObservation],
    validation: EvidenceValidationResult
) -> RiskAssessmentResult:
    """
    Stage 5: Aggregate risk flags from all prior stages.
    This is deterministic — no LLM calls.
    """
    flags = list(qualification.quality_flags)

    # If the object was wrong, flag it.
    if not qualification.object_correct:
        if RiskFlag.WRONG_OBJECT not in flags:
            flags.append(RiskFlag.WRONG_OBJECT)

    # If the claimed part was not visible, flag it.
    if not qualification.claim_part_visible:
        if RiskFlag.WRONG_OBJECT_PART not in flags:
            flags.append(RiskFlag.WRONG_OBJECT_PART)

    # If no damage was visible across any observation, flag it.
    any_damage = any(obs.damage_visible for obs in observations)
    if not any_damage:
        if RiskFlag.DAMAGE_NOT_VISIBLE not in flags:
            flags.append(RiskFlag.DAMAGE_NOT_VISIBLE)

    # If evidence didn't meet the standard but claim wasn't contradicted, flag for review.
    if not validation.evidence_standard_met and validation.claim_status != ClaimStatus.CONTRADICTED:
        if RiskFlag.MANUAL_REVIEW_REQUIRED not in flags:
            flags.append(RiskFlag.MANUAL_REVIEW_REQUIRED)

    # If no flags were generated, explicitly mark none.
    if not flags:
        flags.append(RiskFlag.NONE)

    return RiskAssessmentResult(risk_flags=flags)


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
    # Stage 2: Image Qualification
    # --------------------------------------------------
    # Determine whether the images are suitable for evaluation.
    try:
        qualification = qualify_images(image_paths, claim, evidence_requirement, api_key)
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
    # Aggregate risk flags from all prior stages.
    try:
        risk = _aggregate_risk_flags(qualification, observations, validation)
    except Exception as e:
        print(f"[Pipeline Error] Stage 5 (Risk Assessment) failed: {e}")
        risk = RiskAssessmentResult(risk_flags=[RiskFlag.MANUAL_REVIEW_REQUIRED])

    # --------------------------------------------------
    # Stage 6: Final Output Assembly
    # --------------------------------------------------
    # Assemble all results into the challenge CSV row format.
    severity = _aggregate_severity(observations) if observations else Severity.UNKNOWN

    # Determine the best visible_issue and visible_part from observations
    # for the output row's issue_type and object_part fields.
    if validation.claim_status == ClaimStatus.SUPPORTED and observations:
        # Use the observation that matched the claim.
        matched_obs = next(
            (obs for obs in observations
             if obs.visible_issue == claim.claimed_issue and obs.visible_part == claim.claimed_part),
            None
        )
        issue_type = matched_obs.visible_issue if matched_obs else claim.claimed_issue
        object_part = matched_obs.visible_part.value if matched_obs else claim.claimed_part.value
    else:
        # Fall back to claimed values.
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
