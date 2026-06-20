"""
Groq LLM engine using Groq Cloud API.
Provides high-speed inference for telecom RAG queries.
"""
import logging
import os
from typing import Optional, Generator
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class GroqLLMEngine:
    """Groq LLM engine with API-based inference."""

    def __init__(
        self,
        model_name: str = "llama-3-70b-8192",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ):
        """
        Initialize Groq LLM engine.

        Args:
            model_name: Groq model ID (mixtral-8x7b-32768, llama-3-70b-8192, etc.)
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            max_tokens: Max tokens to generate
            temperature: Generation temperature
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        self.client = Groq(api_key=api_key)
        logger.info(f"✅ Groq LLM engine initialized ({model_name})")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text from prompt via Groq API.

        Args:
            prompt: Input prompt
            max_tokens: Override default max_tokens
            temperature: Override default temperature

        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=0.95,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Generate text with streaming via Groq API.

        Args:
            prompt: Input prompt
            max_tokens: Override default max_tokens
            temperature: Override default temperature

        Yields:
            Generated tokens
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=0.95,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            raise

    def batch_generate(self, prompts: list, **kwargs) -> list:
        """Generate responses for multiple prompts."""
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results

    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = GroqLLMEngine()

    prompt = """You are a telecom engineer. Answer this question:
    What is MIMO in 5G?
    Keep answer brief."""

    response = engine.generate(prompt, max_tokens=100)
    print(f"Response:\n{response}")

    print("\n--- Streaming Example ---")
    for token in engine.generate_streaming(prompt):
        print(token, end="", flush=True)
    print()
