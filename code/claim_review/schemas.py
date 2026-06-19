from enum import Enum
from typing import List, Optional
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

class CarObjectPart(str, Enum):
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

class LaptopObjectPart(str, Enum):
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

class PackageObjectPart(str, Enum):
    BOX = "box"
    PACKAGE_CORNER = "package_corner"
    PACKAGE_SIDE = "package_side"
    SEAL = "seal"
    LABEL = "label"
    CONTENTS = "contents"
    ITEM = "item"
    UNKNOWN = "unknown"

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

class ClaimExtraction(BaseModel):
    """Stage 1: Extract claim details from user conversation."""
    issue_type: IssueType = Field(description="The core issue claimed by the user.")
    object_part: str = Field(description="The part of the object affected. Should map to the relevant enum (Car/Laptop/Package).")
    claim_object: ClaimObject = Field(description="The type of object claimed.")

class ImageQualification(BaseModel):
    """Stage 2: Assess baseline image usability."""
    valid_image: bool = Field(description="True if the image set is usable for automated review.")
    blurry_image: bool = Field(description="True if image is too blurry to use.")
    wrong_object: bool = Field(description="True if image shows a completely different object.")

class InspectionObservation(BaseModel):
    """Stage 3: Extract structured observations strictly from visual evidence."""
    visible_issue_type: IssueType = Field(description="The issue type actually visible in the image.")
    visible_object_part: str = Field(description="The object part actually visible in the image.")
    severity: Severity = Field(description="Estimated severity of the issue.")
    damage_visible: bool = Field(description="True if any damage is clearly visible.")

class EvidenceValidationResult(BaseModel):
    """Stage 4: Validation outcome comparing extraction and observation."""
    evidence_standard_met: bool = Field(description="True if the image set is sufficient to evaluate the claim.")
    evidence_standard_met_reason: str = Field(description="Short reason for the evidence decision.")
    claim_status: ClaimStatus = Field(description="Final decision on the claim.")
    claim_status_justification: str = Field(description="Concise image-grounded explanation.")
    supporting_image_ids: List[str] = Field(description="Image IDs supporting the decision, or ['none'] if none.")

class RiskAssessmentResult(BaseModel):
    """Stage 5: Aggregate risk flags including history."""
    risk_flags: List[RiskFlag] = Field(description="List of detected risk flags.")

class FinalOutputRow(BaseModel):
    """Final Output Row matching the challenge CSV schema exactly."""
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str  # Semicolon-separated strings
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str  # Semicolon-separated strings
    valid_image: bool
    severity: str
