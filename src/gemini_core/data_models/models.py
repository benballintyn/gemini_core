"""
Pydantic models for gemini_core configuration.
"""
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class GeminiConfig(BaseModel):
    """
    Configuration for Gemini generation parameters.
    """

    temperature: Optional[float] = Field(
        default=None, description="Controls the randomness of the output."
    )
    top_p: Optional[float] = Field(
        default=None,
        description="The maximum cumulative probability of tokens to consider when sampling.",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to consider when sampling.",
    )
    candidate_count: Optional[int] = Field(
        default=None, description="Number of generated responses to return."
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to include in a candidate.",
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="The set of character sequences (up to 5) that will stop output generation.",
    )
    response_mime_type: Optional[str] = Field(
        default=None,
        description="Output response mimetype of the generated candidate text.",
    )
    response_schema: Optional[Any] = Field(
        default=None,
        description="Schema for the response. Can be a Pydantic model, dict, or Type.",
    )
    thinking_level: Optional[str] = Field(
        default=None,
        description="Controls the maximum depth of the model's internal reasoning process (low, high).",
    )
    tools: Optional[List[Any]] = Field(
        default=None, description="A list of tools (functions) available to the model."
    )
    tool_config: Optional[Any] = Field(
        default=None, description="Configuration for tool use."
    )
