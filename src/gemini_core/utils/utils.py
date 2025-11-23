"""
Utility functions for gemini_core.
"""
from google.genai import types


def extract_text_from_response(response: types.GenerateContentResponse) -> str:
    """
    Extract the text content from a GenerateContentResponse.

    Args:
        response (types.GenerateContentResponse): The response object.

    Returns:
        str: The extracted text.
    """
    if response.text:
        return response.text
    return ""
