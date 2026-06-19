from typing import List
from .schemas import ClaimExtraction, ImageQualification, InspectionObservation, EvidenceValidationResult, ClaimStatus

def evaluate_claim(
    claim: ClaimExtraction,
    observations: List[InspectionObservation],
    qualification: ImageQualification
) -> EvidenceValidationResult:
    """
    Evaluates a user's damage claim strictly using deterministic rules.
    This function processes visual observations extracted by the LLM, 
    but contains NO LLM calls itself, guaranteeing reproducible logic.
    """
    
    # Rule 1: If no usable image exists -> not_enough_information
    # Rationale: If the image is blurry, corrupted, or otherwise unusable, 
    # we cannot make a safe judgment based on visual evidence.
    if not qualification.valid_image or not qualification.image_usable:
        return EvidenceValidationResult(
            evidence_standard_met=False,
            evidence_standard_met_reason="Image is unusable or invalid.",
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
            claim_status_justification="The provided images are too low quality or invalid to evaluate the claim.",
            supporting_image_ids=["none"]
        )

    # Note: If the image clearly shows the wrong object (e.g., user claims car damage 
    # but uploads a picture of a dog), we contradict. (Optional edge case handled logically).
    if not qualification.object_correct:
        return EvidenceValidationResult(
            evidence_standard_met=True,
            evidence_standard_met_reason="Image is clear but shows the wrong object.",
            claim_status=ClaimStatus.CONTRADICTED,
            claim_status_justification="The submitted image does not show the claimed object.",
            supporting_image_ids=["none"]
        )

    # Rule 2: If claimed part is not visible -> not_enough_information
    # Rationale: We can't confirm or deny damage to a part we cannot see.
    if not qualification.part_visible:
        return EvidenceValidationResult(
            evidence_standard_met=False,
            evidence_standard_met_reason="The claimed object part is not visible.",
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
            claim_status_justification="The image does not show the specific part claimed to be damaged.",
            supporting_image_ids=["none"]
        )

    # If we reach here, we have a usable image, the right object, and the claimed part IS visible.
    # We now evaluate the specific observations.
    
    # We collect all image IDs that support a positive or negative finding.
    supported_image_ids = []
    
    for obs in observations:
        # Rule 3: If visible issue matches claimed issue and claimed part -> supported
        # Rationale: Direct visual confirmation of the user's specific complaint.
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

    # Rule 4: If claimed part is visible and no claimed damage exists -> contradicted
    # Rationale: If the LLM confirmed the part is visible in the image, but asserts that 
    # the issue is 'none' or something unrelated without the claimed damage, the claim is false.
    # We check if across all observations of the visible part, damage is explicitly not there.
    part_visible_no_damage = False
    for obs in observations:
        if obs.part_visible and not obs.damage_visible and obs.visible_issue.value in ["none", "unknown"]:
            part_visible_no_damage = True
            supported_image_ids.append(obs.image_id)
            
    if part_visible_no_damage:
        return EvidenceValidationResult(
            evidence_standard_met=True,
            evidence_standard_met_reason="Claimed part is visible and undamaged.",
            claim_status=ClaimStatus.CONTRADICTED,
            claim_status_justification="The image clearly shows the claimed part, but no damage is present.",
            supporting_image_ids=supported_image_ids
        )

    # Rule 5: If evidence is ambiguous -> not_enough_information
    # Rationale: Catch-all for scenarios where the part is visible, but the issue doesn't exactly match
    # or the LLM's confidence was low, making a definitive ruling unsafe.
    return EvidenceValidationResult(
        evidence_standard_met=False,
        evidence_standard_met_reason="Evidence is ambiguous or inconclusive.",
        claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
        claim_status_justification="The visual evidence is inconclusive regarding the specific claim.",
        supporting_image_ids=["none"]
    )
