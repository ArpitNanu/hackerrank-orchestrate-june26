from typing import List
from .schemas import ClaimExtraction, ImageQualification, InspectionObservation, EvidenceValidationResult, ClaimStatus

MIN_CONFIDENCE = 0.70


def _collect_all_image_ids(observations: List[InspectionObservation]) -> List[str]:
    """
    Collect all unique image IDs from observations, preserving order.
    Used when images were reviewed but evidence was insufficient —
    we still reference which images contributed to the evaluation.
    
    Sample data evidence:
      user_002 (not_enough_info): ids=img_1;img_2 — both images were reviewed
      user_034 (contradicted):    ids=img_1;img_2 — both images contributed
    """
    seen = set()
    ids = []
    for obs in observations:
        if obs.image_id not in seen:
            seen.add(obs.image_id)
            ids.append(obs.image_id)
    return ids if ids else ["none"]


def evaluate_claim(
    claim: ClaimExtraction,
    observations: List[InspectionObservation],
    qualification: ImageQualification
) -> EvidenceValidationResult:
    """
    Evaluates a user's damage claim strictly using deterministic rules.
    This function processes visual observations extracted by the LLM, 
    but contains NO LLM calls itself, guaranteeing reproducible logic.
    
    supporting_image_ids rules:
      SUPPORTED:            IDs of images where the claim was visually confirmed.
      CONTRADICTED:         IDs of images that disprove the claim.
      NOT_ENOUGH_INFO:      IDs of images that were reviewed but insufficient,
                            OR "none" if no inspection was possible at all.
    
    Sample data evidence:
      user_001 (supported):         img_1          — confirming image
      user_003 (supported):         img_2          — specific image, not all
      user_005 (contradicted):      img_1          — contradicting image
      user_002 (not_enough_info):   img_1;img_2    — reviewed but insufficient
      user_006 (not_enough_info):   none           — image couldn't be evaluated
      user_032 (not_enough_info):   none           — image completely unusable
    """
    
    # Rule 1: If no usable image exists -> NOT_ENOUGH_INFORMATION
    # Rationale: If the image is unusable or invalid, no inspection was possible.
    # supporting_image_ids = "none" because no images contributed to evaluation.
    if not qualification.valid_image or not qualification.image_usable:
        return EvidenceValidationResult(
            evidence_standard_met=False,
            evidence_standard_met_reason="Image is unusable or invalid.",
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
            claim_status_justification="The provided images are too low quality or invalid to evaluate the claim.",
            supporting_image_ids=["none"]
        )

    # Rule 2: Wrong object -> NOT_ENOUGH_INFORMATION
    # Rationale: The image may be incorrect, incomplete, or mismatched.
    # supporting_image_ids = all reviewed images (they were inspected, just wrong object).
    # Sample: user_033 (contradicted, ids=img_1) — image was reviewed even though wrong object.
    if not qualification.object_correct:
        return EvidenceValidationResult(
            evidence_standard_met=False,
            evidence_standard_met_reason="Image shows the wrong object.",
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
            claim_status_justification="The submitted image does not show the claimed object.",
            supporting_image_ids=_collect_all_image_ids(observations)
        )

    # Rule 3: If claimed part is not visible -> NOT_ENOUGH_INFORMATION
    # Rationale: We can't confirm or deny damage to a part we cannot see.
    # supporting_image_ids = all reviewed images (they were inspected, part just not visible).
    if not qualification.claim_part_visible:
        return EvidenceValidationResult(
            evidence_standard_met=False,
            evidence_standard_met_reason="The claimed object part is not visible.",
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
            claim_status_justification="The image does not show the specific part claimed to be damaged.",
            supporting_image_ids=_collect_all_image_ids(observations)
        )

    # If we reach here, we have a usable image, the right object, and the claimed part IS visible.
    # We now evaluate the specific observations.
    
    # Filter observations by confidence threshold.
    # Rationale: Ignore observations below threshold to prevent low-confidence LLM guesses from driving decisions.
    valid_observations = [obs for obs in observations if obs.confidence >= MIN_CONFIDENCE]
    
    # Rule 4: If visible issue matches claimed issue and claimed part -> SUPPORTED
    # Rationale: Direct visual confirmation of the user's specific complaint.
    # supporting_image_ids = only the images that confirmed the claim.
    supported_image_ids = []
    for obs in valid_observations:
        if obs.part_visible and obs.visible_issue == claim.claimed_issue and obs.visible_part == claim.claimed_part:
            supported_image_ids.append(obs.image_id)
            
    if supported_image_ids:
        return EvidenceValidationResult(
            evidence_standard_met=True,
            evidence_standard_met_reason="Clear visual evidence of the claimed issue.",
            claim_status=ClaimStatus.SUPPORTED,
            claim_status_justification="Visual evidence matches the claimed issue on the specified part.",
            supporting_image_ids=supported_image_ids
        )

    # Rule 5: Mismatch contradiction -> CONTRADICTED
    # Rationale: Visible evidence disproves the claimed issue (e.g., Claim=scratch, Observation=crack).
    # supporting_image_ids = the images that showed the contradicting evidence.
    mismatch_image_ids = []
    for obs in valid_observations:
        if obs.part_visible and obs.visible_part == claim.claimed_part:
            if obs.visible_issue != claim.claimed_issue and obs.visible_issue.value not in ["none", "unknown"]:
                mismatch_image_ids.append(obs.image_id)
                
    if mismatch_image_ids:
        return EvidenceValidationResult(
            evidence_standard_met=True,
            evidence_standard_met_reason="Claimed part is visible but shows a different issue.",
            claim_status=ClaimStatus.CONTRADICTED,
            claim_status_justification="The visible evidence contradicts the claimed issue type.",
            supporting_image_ids=mismatch_image_ids
        )

    # Rule 6: If claimed part is visible and no claimed damage exists -> CONTRADICTED
    # Rationale: If the part is clearly visible but no damage is present, the claim is false.
    # supporting_image_ids = the images that showed the undamaged part.
    no_damage_image_ids = []
    for obs in valid_observations:
        if obs.part_visible and not obs.damage_visible and obs.visible_issue.value in ["none", "unknown"]:
            no_damage_image_ids.append(obs.image_id)
            
    if no_damage_image_ids:
        return EvidenceValidationResult(
            evidence_standard_met=True,
            evidence_standard_met_reason="Claimed part is visible and undamaged.",
            claim_status=ClaimStatus.CONTRADICTED,
            claim_status_justification="The image clearly shows the claimed part, but no damage is present.",
            supporting_image_ids=no_damage_image_ids
        )

    # Rule 7: If evidence is ambiguous -> NOT_ENOUGH_INFORMATION
    # Rationale: Catch-all for scenarios where the part is visible, but the issue doesn't exactly match
    # or the LLM's confidence was low, making a definitive ruling unsafe.
    # supporting_image_ids = all reviewed images (they were inspected but inconclusive).
    return EvidenceValidationResult(
        evidence_standard_met=False,
        evidence_standard_met_reason="Evidence is ambiguous or inconclusive.",
        claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
        claim_status_justification="The visual evidence is inconclusive regarding the specific claim.",
        supporting_image_ids=_collect_all_image_ids(observations)
    )
