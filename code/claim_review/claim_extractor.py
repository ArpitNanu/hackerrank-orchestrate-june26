import os
from typing import Optional
from openai import OpenAI
from pydantic import ValidationError

from .schemas import ClaimExtraction, ClaimObject, IssueType, CarPart, LaptopPart, PackagePart

# ---------------------------------------------------------
# Prompt Design
# ---------------------------------------------------------
# The prompt enforces strict isolation of responsibility.
# It explicitly forbids the LLM from making judgments or decisions,
# focusing purely on extraction based on allowed schemas.

EXTRACTOR_SYSTEM_PROMPT = """
You are an expert claims extraction assistant. Your job is to read a user's damage claim 
conversation and extract the exact issue type and the specific object part they are complaining about.

RULES:
1. Extract only what the user is claiming.
2. Do not inspect images.
3. Do not determine whether the claim is true.
4. Do not perform risk assessment.
5. Do not generate claim_status.
6. Use the closest matching value for `claimed_issue` and `claimed_part` based on the allowed Enums.
7. If issue cannot be confidently determined: use "unknown".
8. If part cannot be confidently determined: use "unknown".
9. If the user describes functionality problems rather than visible physical damage, prefer UNKNOWN for the issue. Images cannot directly verify functionality-based failures. The extraction stage should not invent physical damage.

EXAMPLES:
User: "My rear bumper has a dent."
Output: claimed_issue = "dent", claimed_part = "rear_bumper"

User: "The laptop screen cracked."
Output: claimed_issue = "crack", claimed_part = "screen"

User: "My package arrived torn."
Output: claimed_issue = "torn_packaging", claimed_part = "box"

User: "My keyboard stopped working."
Output: claimed_issue = "unknown", claimed_part = "keyboard"

User: "The laptop won't turn on."
Output: claimed_issue = "unknown", claimed_part = "unknown"

User: "I think something is wrong with the package."
Output: claimed_issue = "unknown", claimed_part = "unknown"

User: "My battery drains quickly."
Output: claimed_issue = "unknown", claimed_part = "unknown"

User: "The WiFi is not working."
Output: claimed_issue = "unknown", claimed_part = "unknown"
"""

def _get_unknown_part(claim_object: ClaimObject):
    """Returns the correct UNKNOWN enum for the specific claim object."""
    if claim_object == ClaimObject.CAR:
        return CarPart.UNKNOWN
    elif claim_object == ClaimObject.LAPTOP:
        return LaptopPart.UNKNOWN
    elif claim_object == ClaimObject.PACKAGE:
        return PackagePart.UNKNOWN
    return CarPart.UNKNOWN  # Safe fallback

# ---------------------------------------------------------
# Extraction Logic
# ---------------------------------------------------------
# Claim extraction captures user allegations only.
# It does not determine truth.
# Verification happens later in the Evidence Validation stage.
def extract_claim(user_claim: str, claim_object: ClaimObject, api_key: Optional[str] = None) -> ClaimExtraction:
    """
    Extracts structured claim details from the raw user conversation.
    
    Args:
        user_claim: The raw text transcript of the user's claim.
        claim_object: The type of object (car, laptop, package).
        api_key: Optional OpenAI API key.
        
    Returns:
        ClaimExtraction: A strictly typed Pydantic model containing the extracted details.
                         In case of failure, returns a safe fallback with properly typed UNKNOWN values.
    """
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    
    user_prompt = f"Claim Object Type: {claim_object.value}\nUser Claim Transcript:\n{user_claim}"
    
    try:
        # We use the 'parse' method to leverage OpenAI's Structured Outputs (JSON Schema constraint)
        # This guarantees the LLM returns an exact match to our Pydantic model.
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Fast and cheap for simple text extraction
            messages=[
                {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ClaimExtraction,
            temperature=0.0  # Zero temperature for maximum determinism
        )
        
        extracted_claim = response.choices[0].message.parsed
        
        if not extracted_claim:
            raise ValueError("The model failed to return a parsed response.")
            
        return extracted_claim
        
    except ValidationError as e:
        # Error Handling: LLM somehow bypassed schema constraints and hallucinated an invalid Enum.
        print(f"[Error] Validation Error during claim extraction: {e}")
        return ClaimExtraction(
            claim_object=claim_object,
            claimed_issue=IssueType.UNKNOWN,
            claimed_part=_get_unknown_part(claim_object)
        )
        
    except Exception as e:
        # Error Handling: Network error, Rate Limit, or API outage.
        # Fail gracefully by defaulting to unknown, which will trigger manual review downstream.
        print(f"[Error] API Error during claim extraction: {e}")
        return ClaimExtraction(
            claim_object=claim_object,
            claimed_issue=IssueType.UNKNOWN,
            claimed_part=_get_unknown_part(claim_object)
        )
