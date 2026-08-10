"""
Groq LLM engine using Groq Cloud API.
Provides high-speed inference for telecom RAG queries.
"""
import logging
import os
import time
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
        model_name: str = "llama-3.3-70b-versatile",
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

        last_err = None
        for attempt in range(4):
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
                last_err = e
                # Back off and retry on transient/rate-limit errors.
                if "rate" in str(e).lower() or "429" in str(e) or "503" in str(e):
                    wait = 2 ** attempt * 2
                    logger.warning(f"Groq rate/transient error (attempt {attempt+1}/4): retrying in {wait}s")
                    time.sleep(wait)
                    continue
                logger.error(f"Error generating response: {e}")
                raise
        logger.error(f"Groq generation failed after retries: {last_err}")
        raise last_err

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


class LocalLLMEngine:
    """Local inference for a QLoRA-fine-tuned model (base + adapter via peft).

    Use this to run the model fine-tuned in scripts/finetune_qlora_colab.py.
    All heavy deps (torch/transformers/peft) are imported lazily so importing
    this module never requires them when only Groq is used.

    NOTE: A 3B+ model on CPU is slow; for the latency KPI keep GroqLLMEngine as
    the live generator and use this engine for the fine-tuning deliverable
    (ideally on a GPU / Colab).
    """

    def __init__(
        self,
        adapter_path: str,
        base_model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        device: Optional[str] = None,
    ):
        import os as _os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.max_tokens = max_tokens
        self.temperature = temperature

        if base_model is None:
            base_file = _os.path.join(adapter_path, "base_model.txt")
            base_model = open(base_file).read().strip() if _os.path.exists(base_file) else None
        if not base_model:
            raise ValueError("base_model not provided and base_model.txt missing in adapter_path")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading local model {base_model} + adapter {adapter_path} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        self.model_name = f"{base_model}+qlora"
        logger.info("Local QLoRA engine ready")

    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        import torch

        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

    def get_model_info(self) -> dict:
        return {"model": self.model_name, "max_tokens": self.max_tokens, "temperature": self.temperature}


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
