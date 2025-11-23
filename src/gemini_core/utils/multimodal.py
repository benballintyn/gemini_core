"""
Utilities for handling multimodal inputs (images, files).
"""
import mimetypes
from pathlib import Path
from typing import Optional, Union

from google.genai import types


def load_image(
    image: Union[str, Path, bytes], mime_type: Optional[str] = None
) -> types.Part:
    """
    Load an image and create a Part object for Gemini.

    Args:
        image (str | Path | bytes): The image path or bytes.
        mime_type (str, optional): The mime type of the image.
                                   Required if image is bytes.
                                   If None and image is path, guessed from extension.

    Returns:
        types.Part: The image part.

    Raises:
        ValueError: If mime_type is missing for bytes input or cannot be guessed.
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type:
                raise ValueError(
                    f"Could not guess mime type for {path}. Please specify mime_type."
                )

        with open(path, "rb") as f:
            data = f.read()
    else:
        data = image
        if not mime_type:
            raise ValueError("mime_type must be provided when passing bytes.")

    return types.Part.from_bytes(data=data, mime_type=mime_type)
