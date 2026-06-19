import os
from typing import Optional
from openai import OpenAI
from pydantic import ValidationError

from .schemas import ClaimExtraction, ClaimObject

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
1. You must output valid JSON matching the provided schema.
2. Do NOT make policy decisions. Do NOT evaluate the truthfulness of the claim.
3. Only extract what the user is asserting.
4. Use the closest matching value for `issue_type` and `object_part` based on the allowed Enums.
5. If the object part is not explicitly mentioned but implied (e.g. "my screen" on a laptop), map it correctly.
6. If the issue or part cannot be determined from the text, strictly use "unknown".
"""

# ---------------------------------------------------------
# Extraction Logic
# ---------------------------------------------------------

def extract_claim(user_claim: str, claim_object: ClaimObject, api_key: Optional[str] = None) -> ClaimExtraction:
    """
    Extracts structured claim details from the raw user conversation.
    
    Args:
        user_claim: The raw text transcript of the user's claim.
        claim_object: The type of object (car, laptop, package).
        api_key: Optional OpenAI API key.
        
    Returns:
        ClaimExtraction: A strictly typed Pydantic model containing the extracted details.
                         In case of failure, returns a safe fallback with "unknown" values.
    """
    # In the final pipeline, client initialization can be injected or abstracted.
    # We initialize here to maintain module independence as requested.
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    
    user_prompt = f"Claim Object Type: {claim_object.value}\nUser Claim Transcript:\n{user_claim}"
    
    try:
        # We use the 'parse' method to leverage OpenAI's Structured Outputs (JSON Schema constraint)
        # This guarantees the LLM returns an exact match to our Pydantic model.
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Faster and cheaper for simple text extraction tasks
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
            issue_type="unknown",
            object_part="unknown",
            claim_object=claim_object
        )
        
    except Exception as e:
        # Error Handling: Network error, Rate Limit, or API outage.
        # Fail gracefully by defaulting to unknown, which will trigger manual review downstream.
        print(f"[Error] API Error during claim extraction: {e}")
        return ClaimExtraction(
            issue_type="unknown",
            object_part="unknown",
            claim_object=claim_object
        )
