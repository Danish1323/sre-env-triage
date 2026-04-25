"""
Abstract LLM client interface.

Every LLM backend used in this project must implement ``BaseLLMClient``.
This keeps the environment decoupled from any specific provider.

Phase 1 uses OpenRouterLLMClient.
Phase 3+ will add ModalLLMClient for trained model endpoints.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseLLMClient(ABC):
    """
    Minimal interface for all LLM backends.

    Subclasses must implement ``chat_complete`` — a single-turn or
    multi-turn call that returns the assistant's text response.
    """

    @abstractmethod
    def chat_complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        Run a single chat completion request.

        Args:
            system_prompt: The system / instruction prompt.
            user_message:  The user-turn message (current observation).
            temperature:   Sampling temperature.
            max_tokens:    Maximum tokens in the response.

        Returns:
            The model's text response as a string.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
