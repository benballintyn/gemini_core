"""
Main Gemini class implementation.
"""
from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import chats, types
from loguru import logger

from gemini_core.config.config import Config
from gemini_core.data_models.models import GeminiConfig


class Gemini:
    """
    A wrapper class for the Google GenAI SDK to interact with Gemini models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        system_instruction: Optional[str] = None,
        generation_config: Optional[Union[GeminiConfig, Dict[str, Any]]] = None,
    ):
        """
        Initialize the Gemini client.

        Args:
            api_key (str, optional): Google API key. If None, tries to load from environment.
            model_name (str, optional): Name of the model to use. Defaults to config value.
            system_instruction (str, optional): System instruction for the model.
            generation_config (GeminiConfig | Dict, optional): Configuration for generation.
        """
        # Load default config from env if not provided
        try:
            self.config = Config.from_env()
        except ValueError:
            # If loading from env fails (e.g. no API key), we expect api_key to be passed explicitly
            if not api_key:
                raise
            self.config = Config(api_key=api_key)

        # Override config with passed arguments
        if api_key:
            self.config.api_key = api_key
        if model_name:
            self.config.model_name = model_name

        self.client = genai.Client(api_key=self.config.api_key)
        self.system_instruction = system_instruction

        if isinstance(generation_config, GeminiConfig):
            self.generation_config = generation_config
        elif isinstance(generation_config, dict):
            self.generation_config = GeminiConfig(**generation_config)
        else:
            self.generation_config = GeminiConfig()

    def _prepare_config(
        self,
        generation_config: Optional[Union[GeminiConfig, Dict[str, Any]]] = None,
    ) -> types.GenerateContentConfig:
        """
        Prepare the generation config for the SDK.

        Args:
            generation_config (GeminiConfig | Dict, optional): Override default generation config.

        Returns:
            types.GenerateContentConfig: The prepared SDK config.
        """
        config = self.generation_config.model_copy()
        if generation_config:
            if isinstance(generation_config, dict):
                update_data = generation_config
            else:
                update_data = generation_config.model_dump(exclude_unset=True)
            config = config.model_copy(update=update_data)

        # Convert Pydantic model to dict for the SDK, filtering out None values
        gen_config_dict = config.model_dump(exclude_none=True)

        # Auto-set response_mime_type to application/json if response_schema is present
        if (
            "response_schema" in gen_config_dict
            and "response_mime_type" not in gen_config_dict
        ):
            gen_config_dict["response_mime_type"] = "application/json"

        # Handle thinking_level -> thinking_config
        if "thinking_level" in gen_config_dict:
            thinking_level = gen_config_dict.pop("thinking_level")
            if thinking_level:
                gen_config_dict["thinking_config"] = {
                    "thinking_level": thinking_level,
                    "include_thoughts": True,
                }

        # Create GenerateContentConfig
        return types.GenerateContentConfig(
            system_instruction=self.system_instruction, **gen_config_dict
        )

    def generate_content(
        self,
        prompt: Union[str, List[str]],
        generation_config: Optional[Union[GeminiConfig, Dict[str, Any]]] = None,
    ) -> types.GenerateContentResponse:
        """
        Generate content using the Gemini model.

        Args:
            prompt (str | List[str]): The prompt to generate content from.
            generation_config (GeminiConfig | Dict, optional): Override default generation config.

        Returns:
            types.GenerateContentResponse: The generated content response.
        """
        gc_config = self._prepare_config(generation_config)

        logger.debug(f"Generating content with model {self.config.model_name}")

        try:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=gc_config,
            )
            return response
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise

    def upload_file(
        self, path: Union[str, Any], mime_type: Optional[str] = None
    ) -> types.File:
        """
        Upload a file to the File API.

        Args:
            path (str | Path): Path to the file.
            mime_type (str, optional): Mime type of the file.

        Returns:
            types.File: The uploaded file object.
        """
        logger.debug(f"Uploading file: {path}")
        try:
            # If mime_type is provided, we might need to pass it in config
            upload_config = None
            if mime_type:
                upload_config = types.UploadFileConfig(mime_type=mime_type)

            file_obj = self.client.files.upload(file=path, config=upload_config)
            return file_obj
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise

    def start_chat(
        self,
        history: Optional[List[Union[str, types.Content, Dict[str, Any]]]] = None,
        generation_config: Optional[Union[GeminiConfig, Dict[str, Any]]] = None,
    ) -> chats.Chat:
        """
        Start a chat session.

        Args:
            history (List, optional): Initial chat history.
            generation_config (GeminiConfig | Dict, optional): Configuration for generation.

        Returns:
            chats.Chat: The chat session object.
        """
        gc_config = self._prepare_config(generation_config)

        logger.debug(f"Starting chat with model {self.config.model_name}")
        return self.client.chats.create(
            model=self.config.model_name, config=gc_config, history=history
        )

    def count_tokens(self, prompt: Union[str, List[str]]) -> types.CountTokensResponse:
        """
        Count the number of tokens in the prompt.

        Args:
            prompt (str | List[str]): The prompt to count tokens for.

        Returns:
            types.CountTokensResponse: The count tokens response.
        """
        logger.debug(f"Counting tokens for model {self.config.model_name}")
        try:
            response = self.client.models.count_tokens(
                model=self.config.model_name,
                contents=prompt,
            )
            return response
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            raise

    def generate_content_stream(
        self,
        prompt: Union[str, List[str]],
        generation_config: Optional[Union[GeminiConfig, Dict[str, Any]]] = None,
    ) -> Any:
        """
        Generate content stream using the Gemini model.

        Args:
            prompt (str | List[str]): The prompt to generate content from.
            generation_config (GeminiConfig | Dict, optional): Override default generation config.

        Yields:
            types.GenerateContentResponse: Chunks of the generated content.
        """
        gc_config = self._prepare_config(generation_config)

        logger.debug(f"Generating content stream with model {self.config.model_name}")

        try:
            response = self.client.models.generate_content_stream(
                model=self.config.model_name,
                contents=prompt,
                config=gc_config,
            )
            for chunk in response:
                yield chunk
        except Exception as e:
            logger.error(f"Error generating content stream: {e}")
            raise

    async def generate_content_async(
        self,
        prompt: Union[str, List[str]],
        generation_config: Optional[Union[GeminiConfig, Dict[str, Any]]] = None,
    ) -> types.GenerateContentResponse:
        """
        Generate content asynchronously using the Gemini model.

        Args:
            prompt (str | List[str]): The prompt to generate content from.
            generation_config (GeminiConfig | Dict, optional): Override default generation config.

        Returns:
            types.GenerateContentResponse: The generated content response.
        """
        gc_config = self._prepare_config(generation_config)

        logger.debug(f"Generating content async with model {self.config.model_name}")

        try:
            response = await self.client.aio.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=gc_config,
            )
            return response
        except Exception as e:
            logger.error(f"Error generating content async: {e}")
            raise

    async def generate_content_stream_async(
        self,
        prompt: Union[str, List[str]],
        generation_config: Optional[Union[GeminiConfig, Dict[str, Any]]] = None,
    ) -> Any:
        """
        Generate content stream asynchronously using the Gemini model.

        Args:
            prompt (str | List[str]): The prompt to generate content from.
            generation_config (GeminiConfig | Dict, optional): Override default generation config.

        Yields:
            types.GenerateContentResponse: Chunks of the generated content.
        """
        gc_config = self._prepare_config(generation_config)

        logger.debug(
            f"Generating content stream async with model {self.config.model_name}"
        )

        try:
            async for chunk in self.client.aio.models.generate_content_stream(
                model=self.config.model_name,
                contents=prompt,
                config=gc_config,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error generating content stream async: {e}")
            raise

    async def count_tokens_async(
        self, prompt: Union[str, List[str]]
    ) -> types.CountTokensResponse:
        """
        Count the number of tokens in the prompt asynchronously.

        Args:
            prompt (str | List[str]): The prompt to count tokens for.

        Returns:
            types.CountTokensResponse: The count tokens response.
        """
        logger.debug(f"Counting tokens async for model {self.config.model_name}")
        try:
            response = await self.client.aio.models.count_tokens(
                model=self.config.model_name,
                contents=prompt,
            )
            return response
        except Exception as e:
            logger.error(f"Error counting tokens async: {e}")
            raise

    async def upload_file_async(
        self, path: Union[str, Any], mime_type: Optional[str] = None
    ) -> types.File:
        """
        Upload a file to the File API asynchronously.

        Args:
            path (str | Path): Path to the file.
            mime_type (str, optional): Mime type of the file.

        Returns:
            types.File: The uploaded file object.
        """
        logger.debug(f"Uploading file async: {path}")
        try:
            # If mime_type is provided, we might need to pass it in config
            upload_config = None
            if mime_type:
                upload_config = types.UploadFileConfig(mime_type=mime_type)

            file_obj = await self.client.aio.files.upload(
                file=path, config=upload_config
            )
            return file_obj
        except Exception as e:
            logger.error(f"Error uploading file async: {e}")
            raise

    async def start_chat_async(
        self,
        history: Optional[List[Union[str, types.Content, Dict[str, Any]]]] = None,
        generation_config: Optional[Union[GeminiConfig, Dict[str, Any]]] = None,
    ) -> chats.AsyncChat:
        """
        Start an async chat session.

        Args:
            history (List, optional): Initial chat history.
            generation_config (GeminiConfig | Dict, optional): Configuration for generation.

        Returns:
            chats.AsyncChat: The async chat session object.
        """
        gc_config = self._prepare_config(generation_config)

        logger.debug(f"Starting async chat with model {self.config.model_name}")
        return self.client.aio.chats.create(
            model=self.config.model_name, config=gc_config, history=history
        )
