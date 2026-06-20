from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Enums
# ---------------------------------------------------------

class IssueType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    BROKEN_PART = "broken_part"
    MISSING_PART = "missing_part"
    TORN_PACKAGING = "torn_packaging"
    CRUSHED_PACKAGING = "crushed_packaging"
    WATER_DAMAGE = "water_damage"
    STAIN = "stain"
    NONE = "none"
    UNKNOWN = "unknown"

class ClaimObject(str, Enum):
    CAR = "car"
    LAPTOP = "laptop"
    PACKAGE = "package"

class CarPart(str, Enum):
    FRONT_BUMPER = "front_bumper"
    REAR_BUMPER = "rear_bumper"
    DOOR = "door"
    HOOD = "hood"
    WINDSHIELD = "windshield"
    SIDE_MIRROR = "side_mirror"
    HEADLIGHT = "headlight"
    TAILLIGHT = "taillight"
    FENDER = "fender"
    QUARTER_PANEL = "quarter_panel"
    BODY = "body"
    UNKNOWN = "unknown"

class LaptopPart(str, Enum):
    SCREEN = "screen"
    KEYBOARD = "keyboard"
    TRACKPAD = "trackpad"
    HINGE = "hinge"
    LID = "lid"
    CORNER = "corner"
    PORT = "port"
    BASE = "base"
    BODY = "body"
    UNKNOWN = "unknown"

class PackagePart(str, Enum):
    BOX = "box"
    PACKAGE_CORNER = "package_corner"
    PACKAGE_SIDE = "package_side"
    SEAL = "seal"
    LABEL = "label"
    CONTENTS = "contents"
    ITEM = "item"
    UNKNOWN = "unknown"

ObjectPart = Union[CarPart, LaptopPart, PackagePart]

class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"

class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"

class RiskFlag(str, Enum):
    NONE = "none"
    BLURRY_IMAGE = "blurry_image"
    CROPPED_OR_OBSTRUCTED = "cropped_or_obstructed"
    LOW_LIGHT_OR_GLARE = "low_light_or_glare"
    WRONG_ANGLE = "wrong_angle"
    WRONG_OBJECT = "wrong_object"
    WRONG_OBJECT_PART = "wrong_object_part"
    DAMAGE_NOT_VISIBLE = "damage_not_visible"
    CLAIM_MISMATCH = "claim_mismatch"
    POSSIBLE_MANIPULATION = "possible_manipulation"
    NON_ORIGINAL_IMAGE = "non_original_image"
    TEXT_INSTRUCTION_PRESENT = "text_instruction_present"
    USER_HISTORY_RISK = "user_history_risk"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

class UserHistoryRow(BaseModel):
    """Structured representation of a user's claim history."""
    past_claim_count: int = 0
    accept_claim: int = 0
    manual_review_claim: int = 0
    rejected_claim: int = 0
    last_90_days_claim_count: int = 0
    history_flags: str = ""
    history_summary: str = ""

class ClaimExtraction(BaseModel):
    """Stage 1: Extract claim details from user conversation."""
    claim_object: ClaimObject = Field(description="The type of object claimed.")
    claimed_issue: IssueType = Field(description="The core issue claimed by the user.")
    claimed_part: ObjectPart = Field(description="The part of the object affected.")

class ImageQualification(BaseModel):
    """Stage 2: Assess baseline image usability."""
    valid_image: bool = Field(description="True if the image set is usable for automated review.")
    image_usable: bool = Field(description="True if the image is clear enough to inspect.")
    object_correct: bool = Field(description="True if the image shows the correct object type.")
    part_visible: bool = Field(description="True if the claimed object part is visible.")
    claim_part_visible: bool = Field(
        description="True if the specific object part referenced by the claim is visible in at least one usable image."
    )
    quality_flags: List[RiskFlag] = Field(
        default_factory=list,
        description="Any detected image quality risks."
    )

class InspectionObservation(BaseModel):
    """Stage 3: Extract structured observations strictly from visual evidence."""
    image_id: str = Field(description="The ID of the image being observed.")
    visible_issue: IssueType = Field(description="The actual issue type visible in the image.")
    visible_part: ObjectPart = Field(description="The actual object part visible in the image.")
    severity: Severity = Field(description="Estimated severity of the issue.")
    damage_visible: bool = Field(description="True if any damage is clearly visible.")
    part_visible: bool = Field(description="True if the specific part is clearly visible in this image.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score of the observation (0.0 to 1.0)."
    )
    observation_reason: str = Field(
        description="Short image-grounded explanation of what was visually observed."
    )

class EvidenceValidationResult(BaseModel):
    """Stage 4: Validation outcome comparing extraction and observation."""
    evidence_standard_met: bool = Field(description="True if the image set is sufficient to evaluate the claim.")
    evidence_standard_met_reason: str = Field(description="Short reason for the evidence decision.")
    claim_status: ClaimStatus = Field(description="Final decision on the claim.")
    claim_status_justification: str = Field(description="Concise image-grounded explanation.")
    supporting_image_ids: List[str] = Field(
        default_factory=list,
        description="Image IDs supporting the decision, or ['none'] if none."
    )

class RiskAssessmentResult(BaseModel):
    """Stage 5: Aggregate risk flags including history."""
    risk_flags: List[RiskFlag] = Field(
        default_factory=list,
        description="List of detected risk flags."
    )

class FinalOutputRow(BaseModel):
    """Final Output Row matching the challenge CSV schema exactly."""
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str  # Semicolon-separated strings
    issue_type: IssueType
    object_part: str
    claim_status: ClaimStatus
    claim_status_justification: str
    supporting_image_ids: str  # Semicolon-separated strings
    valid_image: bool
    severity: Severity

class RequirementResolution(BaseModel):
    """The resolved minimum image evidence requirement."""
    requirement_id: str
    requirement_text: str
