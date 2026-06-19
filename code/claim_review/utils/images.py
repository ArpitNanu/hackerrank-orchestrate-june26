import os
import base64


def encode_image(image_path: str) -> str:
    """Base64 encodes an image file for the OpenAI Vision API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime_type(path: str) -> str:
    """Infers MIME type from file extension."""
    lower = path.lower()
    if lower.endswith(".png"):
        return "image/png"
    elif lower.endswith(".webp"):
        return "image/webp"
    elif lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"
