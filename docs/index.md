# gemini_core

A robust, feature-rich, and easy-to-use Python wrapper for the Google GenAI SDK (`google-genai`). This package simplifies interaction with Gemini models, providing a unified interface for synchronous, asynchronous, and streaming generation, along with advanced features like multimodal inputs, function calling, and thought signatures (Gemini 3).

## Installation

```bash
pip install gemini-core
# or with poetry
poetry add gemini-core
```

## Configuration

You can configure the client using environment variables or by passing arguments directly.

**Environment Variables (.env):**
```env
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3-pro-preview  # Optional, defaults to gemini-3-pro-preview
GEMINI_PROJECT_ID=your_project_id  # Optional
GEMINI_LOCATION=us-central1       # Optional
```

## Usage

### Basic Generation

```python
from gemini_core import Gemini

client = Gemini() # Loads API key from env
response = client.generate_content("Tell me a joke.")
print(response.text)
```

### Async & Streaming

```python
import asyncio
from gemini_core import Gemini

async def main():
    client = Gemini()
    
    # Async
    response = await client.generate_content_async("Hello async world!")
    print(response.text)

    # Streaming (Sync)
    for chunk in client.generate_content_stream("Tell me a story"):
        print(chunk.text, end="")

    # Streaming (Async)
    async for chunk in client.generate_content_stream_async("Tell me a long story"):
        print(chunk.text, end="")

if __name__ == "__main__":
    asyncio.run(main())
```

### Chat (Multi-turn)

The `start_chat` method handles history and thought signatures automatically.

```python
client = Gemini()
chat = client.start_chat(history=["Hello, I'm a bot."])
response = chat.send_message("Hi there!")
print(response.text)
```

### Structured Output

Pass a Pydantic model to `response_schema` to get structured JSON output.

```python
from pydantic import BaseModel
from gemini_core import Gemini, GeminiConfig

class Recipe(BaseModel):
    name: str
    ingredients: list[str]

client = Gemini()
config = GeminiConfig(response_schema=Recipe)
response = client.generate_content("Cookie recipe", generation_config=config)
print(response.text) # JSON string matching Recipe schema
```

### Multimodal (Images & Files)

```python
from gemini_core import Gemini
from gemini_core.utils.multimodal import load_image

client = Gemini()

# Inline Image
image_part = load_image("path/to/image.jpg")
response = client.generate_content(["Describe this image", image_part])

# File Upload (for large files/video)
file_obj = client.upload_file("path/to/video.mp4")
response = client.generate_content(["Summarize this video", file_obj])
```

### Function Calling (Tools)

```python
from gemini_core import Gemini, GeminiConfig

def get_weather(location: str):
    return f"The weather in {location} is sunny."

client = Gemini()
config = GeminiConfig(tools=[get_weather])

# The SDK handles tool execution automatically
response = client.generate_content("What's the weather in Tokyo?", generation_config=config)
print(response.text)
```

### Thinking (Gemini 3)

Enable "thinking" mode for deeper reasoning.

```python
from gemini_core import Gemini, GeminiConfig

client = Gemini()
config = GeminiConfig(thinking_level="high")
response = client.generate_content("Solve this complex logic puzzle...", generation_config=config)
print(response.text)
```

## Development

This project uses `poetry` for dependency management.

```bash
poetry install
poetry run pytest
```
