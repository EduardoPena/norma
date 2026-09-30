"""LLM client factory and implementations"""

from abc import ABC, abstractmethod
from typing import Optional
import os

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class EmptyModelContentError(RuntimeError):
    """Raised when a provider returns a completion without usable content."""


def _empty_content_error(response) -> EmptyModelContentError:
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None)
    return EmptyModelContentError(
        "Model returned empty content "
        f"(finish_reason={choice.finish_reason!r}, "
        f"completion_tokens={completion_tokens}, "
        f"reasoning_tokens={reasoning_tokens})"
    )


def _extract_openai_compatible_text(response) -> str:
    """Extract text safely from OpenAI-compatible chat responses."""
    message = response.choices[0].message
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(text)
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(text)

        merged = "\n".join(part for part in parts if part).strip()
        if merged:
            return merged

    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        return refusal.strip()

    raise _empty_content_error(response)


class LLMClient(ABC):
    """Abstract base for LLM clients"""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """
        Generate completion for a prompt

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        pass


class OpenAIClient(LLMClient):
    """OpenAI GPT client"""

    def __init__(self, api_key: str, model_name: str = "gpt-5-mini"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        # GPT-5 and some models only support temperature=1
        self.temperature = 1 if model_name.startswith("gpt-5") else 0.0

    def complete(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return _extract_openai_compatible_text(response)
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e


class MaritacaAIClient(LLMClient):
    """Maritaca client"""

    def __init__(self, api_key: str, model_name: str = "sabiazinho-4"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url="https://chat.maritaca.ai/api")
        self.model_name = model_name
        # GPT-5 and some models only support temperature=1
        self.temperature = 1 if model_name.startswith("sabia") else 0.0

    def complete(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return _extract_openai_compatible_text(response)
        except Exception as e:
            raise RuntimeError(f"Maritaca API error: {e}") from e


class AnthropicClient(LLMClient):
    """Anthropic Claude client"""

    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20241022"):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def complete(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}") from e


class MistralClient(LLMClient):
    """Mistral AI client"""

    def __init__(self, api_key: str, model_name: str = "mistral-small-2506"):
        try:
            from mistralai import Mistral
        except ImportError as e:
            raise ImportError(
                "mistralai package not installed. Install with: pip install mistralai"
            ) from e

        self.client = Mistral(api_key=api_key)
        self.model_name = model_name

    def complete(self, prompt: str) -> str:
        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return _extract_openai_compatible_text(response)
        except Exception as e:
            raise RuntimeError(f"Mistral API error: {e}") from e


class GroqClient(LLMClient):
    """Groq client (Llama, Mixtral, Gemma, Qwen, etc.)"""

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model_name = model_name

    def complete(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return _extract_openai_compatible_text(response)
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}") from e


class OpenRouterClient(LLMClient):
    """OpenRouter client using the OpenAI-compatible API."""

    def __init__(self, api_key: str, model_name: str, max_tokens: Optional[int] = None):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = 1 if model_name.startswith("gpt-5") else 0.0

    def complete(self, prompt: str) -> str:
        try:
            request = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            if self.max_tokens is not None:
                request["max_tokens"] = self.max_tokens
            for attempt in range(3):
                response = self.client.chat.completions.create(**request)
                try:
                    return _extract_openai_compatible_text(response)
                except EmptyModelContentError as error:
                    if attempt == 2:
                        raise
                    print(
                        f"OpenRouter returned empty content; retrying completion: {error}"
                    )
        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {e}") from e


def _get_required_api_key(env_var: str) -> str:
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"{env_var} environment variable not set")
    return api_key


def _is_openrouter_model(model_name: str) -> bool:
    return model_name.startswith("openrouter/") or model_name.startswith("openrouter:")


def _normalize_openrouter_model_name(model_name: str) -> str:
    if model_name.startswith("openrouter/"):
        return model_name.split("/", 1)[1]
    if model_name.startswith("openrouter:"):
        return model_name.split(":", 1)[1]
    return model_name


def _get_openrouter_client(
    model_name: str, max_tokens: Optional[int] = None
) -> OpenRouterClient:
    api_key = _get_required_api_key("OPENROUTER_API_KEY")
    return OpenRouterClient(
        api_key=api_key,
        model_name=_normalize_openrouter_model_name(model_name),
        max_tokens=max_tokens,
    )


def get_llm_client(
    model_name: str = "gpt-5-mini",
    provider: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> LLMClient:
    """
    Factory to create LLM clients based on model name

    Args:
        model_name: Model identifier (e.g., "gpt-5-mini", "claude-3-5-sonnet-20241022")
        provider: Optional explicit provider override. Supported: openai,
            openrouter, maritaca, anthropic, mistral, groq.

    Returns:
        LLMClient instance

    Raises:
        ValueError: If API key is missing or model is unknown
    """
    provider = provider.lower() if provider else None

    if provider == "openrouter" or _is_openrouter_model(model_name):
        return _get_openrouter_client(model_name, max_tokens=max_tokens)

    if provider == "openai":
        return OpenAIClient(
            api_key=_get_required_api_key("OPENAI_API_KEY"),
            model_name=model_name,
        )

    if provider == "maritaca":
        return MaritacaAIClient(
            api_key=_get_required_api_key("MARITACA_API_KEY"),
            model_name=model_name,
        )

    if provider == "anthropic":
        return AnthropicClient(
            api_key=_get_required_api_key("ANTHROPIC_API_KEY"),
            model_name=model_name,
        )

    if provider == "mistral":
        return MistralClient(
            api_key=_get_required_api_key("MISTRAL_API_KEY"),
            model_name=model_name,
        )

    if provider == "groq":
        return GroqClient(
            api_key=_get_required_api_key("GROQ_API_KEY"),
            model_name=model_name,
        )

    if provider is not None:
        raise ValueError(f"Unknown provider: {provider}")

    if model_name.startswith("gpt-"):
        return OpenAIClient(
            api_key=_get_required_api_key("OPENAI_API_KEY"),
            model_name=model_name,
        )

    if model_name.startswith("sabia"):
        return MaritacaAIClient(
            api_key=_get_required_api_key("MARITACA_API_KEY"),
            model_name=model_name,
        )

    if model_name.startswith("claude-"):
        return AnthropicClient(
            api_key=_get_required_api_key("ANTHROPIC_API_KEY"),
            model_name=model_name,
        )

    if model_name.startswith("mistral-"):
        return MistralClient(
            api_key=_get_required_api_key("MISTRAL_API_KEY"),
            model_name=model_name,
        )

    if model_name.startswith(("llama-", "moonshotai", "qwen", "gemma", "mixtral")):
        return GroqClient(
            api_key=_get_required_api_key("GROQ_API_KEY"),
            model_name=model_name,
        )

    if os.getenv("OPENROUTER_API_KEY"):
        return _get_openrouter_client(model_name, max_tokens=max_tokens)

    raise ValueError(
        f"Unknown model: {model_name}. Set OPENROUTER_API_KEY to route arbitrary models via OpenRouter."
    )
