from pydantic import BaseModel


class Property(BaseModel):
    """
    A simple key-value pair property.
    Used to represent properties in a way that avoids 'additionalProperties' issues with Gemini API.
    """

    key: str
    value: str
