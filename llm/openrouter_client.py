"""
OpenRouter LLM client — Phase 1 backend.

Uses the OpenAI-compatible API exposed by OpenRouter, with the paid model:
    meta-llama/llama-3.2-3b-instruct

Set your key in the .env file:
    OPENROUTER_API_KEY=<your key>

Usage::

    from llm.openrouter_client import OpenRouterLLMClient
    client = OpenRouterLLMClient()
    response = client.chat_complete(
        system_prompt="You are an SRE agent.",
        user_message="Observation: ...",
    )
"""

import logging
import os
from typing import Optional

from openai import OpenAI

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/llama-3.2-3b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLLMClient(BaseLLMClient):
    """
    LLM client that calls OpenRouter using the OpenAI SDK.

    Args:
        model:   OpenRouter model string (default: meta-llama/llama-3.2-3b-instruct).
        api_key: OpenRouter API key. Reads OPENROUTER_API_KEY env var if not provided.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        _key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not _key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file or pass api_key= explicitly."
            )
        self._client = OpenAI(
            api_key=_key,
            base_url=OPENROUTER_BASE_URL,
        )
        logger.info("OpenRouterLLMClient initialised with model=%s", self.model)

    def chat_complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        Run a chat completion via OpenRouter.

        Args:
            system_prompt: Instruction / persona for the model.
            user_message:  Current observation or context.
            temperature:   Sampling temperature (0 = deterministic).
            max_tokens:    Maximum tokens to generate.

        Returns:
            The model's text reply as a plain string.

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            logger.debug("OpenRouter response: %s", text[:200])
            return text.strip()
        except Exception as exc:
            logger.error("OpenRouter API call failed: %s", exc)
            raise RuntimeError(f"OpenRouter call failed: {exc}") from exc

    def __repr__(self) -> str:
        return f"OpenRouterLLMClient(model={self.model!r})"
