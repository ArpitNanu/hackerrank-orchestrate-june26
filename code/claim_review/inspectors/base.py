import os
from abc import ABC, abstractmethod
from typing import List, Optional
from openai import OpenAI
from pydantic import ValidationError

from ..schemas import InspectionObservation, ClaimObject, IssueType, Severity
from ..utils.images import encode_image, get_mime_type, extract_image_id


class BaseInspector(ABC):
    """
    Shared interface for all object-specific inspectors.
    
    Inspectors examine images and produce structured visual observations.
    They answer ONLY: "What is visually observable?"
    
    Inspectors do NOT:
    - determine claim status
    - perform risk assessment
    - inspect user history
    - decide supported/contradicted/not_enough_information
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Each inspector provides its own domain-specific system prompt."""
        ...

    @property
    @abstractmethod
    def claim_object(self) -> ClaimObject:
        """The object type this inspector handles."""
        ...

    def inspect(self, image_paths: List[str]) -> List[InspectionObservation]:
        """
        Inspects each image independently and returns one observation per image.
        
        Args:
            image_paths: List of paths to the submitted images.
            
        Returns:
            List[InspectionObservation]: One observation per successfully loaded image.
        """
        observations = []

        for path in image_paths:
            image_id = extract_image_id(path)

            if not os.path.exists(path):
                print(f"[Warning] Image path not found: {path}")
                observations.append(self._fallback_observation(image_id))
                continue

            try:
                observation = self._inspect_single_image(path, image_id)
                observations.append(observation)
            except Exception as e:
                print(f"[Error] Failed to inspect {path}: {e}")
                observations.append(self._fallback_observation(image_id))

        return observations

    def _inspect_single_image(self, image_path: str, image_id: str) -> InspectionObservation:
        """
        Sends a single image to the LLM for structured visual observation.
        """
        base64_image = encode_image(image_path)
        mime_type = get_mime_type(image_path)

        content_payload = [
            {"type": "text", "text": f"Image ID: {image_id}\nInspect this image and report your visual observations."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            }
        ]

        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content_payload}
            ],
            response_format=InspectionObservation,
            temperature=0.0
        )

        observation = response.choices[0].message.parsed

        if not observation:
            raise ValueError(f"Model failed to return a parsed observation for {image_id}.")

        # Ensure the image_id is correctly set (LLM may hallucinate a different one).
        observation.image_id = image_id

        return observation

    def _fallback_observation(self, image_id: str) -> InspectionObservation:
        """
        Returns a safe fallback observation when an image cannot be processed.
        Uses UNKNOWN values so the deterministic rules engine can handle it.
        """
        return InspectionObservation(
            image_id=image_id,
            visible_issue=IssueType.UNKNOWN,
            visible_part=self._get_unknown_part(),
            severity=Severity.UNKNOWN,
            damage_visible=False,
            part_visible=False,
            confidence=0.0,
            observation_reason="Image could not be processed."
        )

    @abstractmethod
    def _get_unknown_part(self):
        """Returns the correct UNKNOWN part enum for this inspector's object type."""
        ...
