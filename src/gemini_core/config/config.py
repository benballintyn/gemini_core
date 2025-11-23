"""
Configuration management for gemini_core.
"""
import os
from dataclasses import dataclass
from typing import Optional

from cogito.utils.config import load_envs


@dataclass
class Config:
    """
    Configuration for the Gemini wrapper.
    """

    api_key: str
    model_name: str = "gemini-3-pro-preview"
    project_id: Optional[str] = None
    location: Optional[str] = None

    @classmethod
    def from_env(cls, package_dir: Optional[str] = None) -> "Config":
        """
        Load configuration from environment variables.

        Args:
            package_dir (str, optional): Directory to load .env files from.
                                         Defaults to the current working directory.

        Returns:
            Config: The configuration object.

        Raises:
            ValueError: If GOOGLE_API_KEY is not set.
        """
        if package_dir:
            load_envs(package_dir)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        return cls(
            api_key=api_key,
            model_name=os.getenv("GEMINI_MODEL", "gemini-3-pro-preview"),
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        )
