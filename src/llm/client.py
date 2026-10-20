"""LLM client factory and implementations"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, Optional, Set
import json
import os
import urllib.request

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class EmptyModelContentError(RuntimeError):
    """Raised when a provider returns a completion without usable content.

    This is a model-output failure (e.g. reasoning exhausted max_tokens), not an
    infrastructure failure, so callers may treat the item as unanswered.
    """


class ProviderResponseError(RuntimeError):
    """A 200 response without choices (upstream provider failure); retryable."""


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
    def complete(
        self,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "output",
    ) -> str:
        """
        Generate completion for a prompt

        Args:
            prompt: Input prompt
            response_schema: Optional JSON Schema (src.llm.schemas) for structured
                output. Clients without structured-output support ignore it; the
                prompts describe the same JSON format.
            schema_name: Name sent with the schema

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

    def complete(self, prompt: str, response_schema=None, schema_name="output") -> str:
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

    def complete(self, prompt: str, response_schema=None, schema_name="output") -> str:
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

    def complete(self, prompt: str, response_schema=None, schema_name="output") -> str:
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

    def complete(self, prompt: str, response_schema=None, schema_name="output") -> str:
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

    def complete(self, prompt: str, response_schema=None, schema_name="output") -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return _extract_openai_compatible_text(response)
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}") from e


def _is_transient_api_error(error: Exception) -> bool:
    """Rate limits, timeouts, connection drops, 5xx and empty provider responses are retryable."""
    from openai import APIConnectionError, APIStatusError

    if isinstance(error, (APIConnectionError, ProviderResponseError)):
        return True  # APIConnectionError includes APITimeoutError
    if isinstance(error, APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    return False


@lru_cache(maxsize=1)
def _openrouter_supported_parameters() -> Dict[str, Set[str]]:
    """Model id -> request parameters accepted by its OpenRouter endpoints."""
    with urllib.request.urlopen(f"{OPENROUTER_BASE_URL}/models", timeout=30) as response:
        return {
            model["id"]: set(model.get("supported_parameters") or [])
            for model in json.load(response)["data"]
        }


class OpenRouterClient(LLMClient):
    """OpenRouter client using the OpenAI-compatible API.

    Only parameters the model supports are sent (e.g. GPT-5 rejects
    ``temperature``), which also keeps ``provider.require_parameters`` routable.
    """

    # Exponential backoff for transient API errors: 5, 10, ..., 160, then 300 s
    # (~25 min in total; upstream rate limits on shared pools can last minutes).
    max_transient_retries = 10
    backoff_base_s = 5.0
    backoff_max_s = 300.0

    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_tokens: Optional[int] = None,
        structured_output: bool = True,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = 0.0
        # Optional sampling seed (NORMA_LLM_SEED), sent only to models whose catalog
        # entry lists "seed": reduces run-to-run variance where providers honor it.
        # Opt-in because require_parameters would also restrict routing to
        # providers that accept a seed.
        seed = os.environ.get("NORMA_LLM_SEED", "").strip()
        self.seed = int(seed) if seed else None
        try:
            self.supported_parameters = _openrouter_supported_parameters().get(model_name)
        except Exception as error:  # catalog unavailable: send parameters as before
            print(f"Warning: could not read OpenRouter model catalog: {error}")
            self.supported_parameters = None
        self.structured_output = structured_output and self._supports("response_format")
        # Optional JSON-mode fallback (NORMA_JSON_OBJECT_MODELS, comma-separated
        # model ids) for models whose strict json_schema decoding is unreliable:
        # the schema goes into the prompt, response_format is json_object, and
        # callers validate the output as they do for prompt-only JSON.
        json_object_models = {
            model.strip()
            for model in os.environ.get("NORMA_JSON_OBJECT_MODELS", "").split(",")
            if model.strip()
        }
        self.json_object_mode = self.structured_output and model_name in json_object_models
        if structured_output and not self.structured_output:
            print(
                f"Warning: {model_name} does not support response_format; "
                "using prompt-only JSON output"
            )

    def _supports(self, parameter: str) -> bool:
        return self.supported_parameters is None or parameter in self.supported_parameters

    def _create_with_backoff(self, request: dict):
        import time

        for retry in range(self.max_transient_retries + 1):
            try:
                response = self.client.chat.completions.create(**request)
                if not getattr(response, "choices", None):
                    raise ProviderResponseError(
                        "Provider returned no choices: "
                        f"{getattr(response, 'error', None) or 'no error detail'}"
                    )
                return response
            except Exception as error:
                if retry == self.max_transient_retries or not _is_transient_api_error(
                    error
                ):
                    raise
                delay = min(self.backoff_base_s * 2**retry, self.backoff_max_s)
                print(
                    f"OpenRouter transient error ({type(error).__name__}, "
                    f"status={getattr(error, 'status_code', None)}) for "
                    f"{self.model_name}; retry {retry + 1}/"
                    f"{self.max_transient_retries} in {delay:.0f}s",
                    flush=True,
                )
                time.sleep(delay)

    def complete(
        self,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "output",
    ) -> str:
        try:
            request = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self._supports("temperature"):
                request["temperature"] = self.temperature
            if self.max_tokens is not None and self._supports("max_tokens"):
                request["max_tokens"] = self.max_tokens
            seed = getattr(self, "seed", None)
            if seed is not None and "seed" in (self.supported_parameters or ()):
                request["seed"] = seed
            if response_schema is not None and getattr(self, "json_object_mode", False):
                request["messages"][0]["content"] = (
                    f"{prompt}\n\nRespond with ONLY a JSON object that matches this "
                    f"JSON schema:\n{json.dumps(response_schema)}"
                )
                request["response_format"] = {"type": "json_object"}
                request["extra_body"] = {"provider": {"require_parameters": True}}
            elif response_schema is not None and self.structured_output:
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": response_schema,
                    },
                }
                # Route only to providers that honor response_format.
                request["extra_body"] = {"provider": {"require_parameters": True}}
            for attempt in range(3):
                response = self._create_with_backoff(request)
                try:
                    return _extract_openai_compatible_text(response)
                except EmptyModelContentError as error:
                    if attempt == 2:
                        raise
                    print(
                        f"OpenRouter returned empty content; retrying completion: {error}"
                    )
        except EmptyModelContentError:
            raise
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
    model_name: str, max_tokens: Optional[int] = None, structured_output: bool = True
) -> OpenRouterClient:
    api_key = _get_required_api_key("OPENROUTER_API_KEY")
    return OpenRouterClient(
        api_key=api_key,
        model_name=_normalize_openrouter_model_name(model_name),
        max_tokens=max_tokens,
        structured_output=structured_output,
    )


def get_llm_client(
    model_name: str = "gpt-5-mini",
    provider: Optional[str] = None,
    max_tokens: Optional[int] = None,
    structured_output: bool = True,
) -> LLMClient:
    """
    Factory to create LLM clients based on model name

    Args:
        model_name: Model identifier (e.g., "gpt-5-mini", "claude-3-5-sonnet-20241022")
        provider: Optional explicit provider override. Supported: openai,
            openrouter, maritaca, anthropic, mistral, groq.
        structured_output: Send JSON Schemas as strict structured output
            (OpenRouter only; other clients rely on the prompt format).

    Returns:
        LLMClient instance

    Raises:
        ValueError: If API key is missing or model is unknown
    """
    provider = provider.lower() if provider else None

    if provider == "openrouter" or _is_openrouter_model(model_name):
        return _get_openrouter_client(
            model_name, max_tokens=max_tokens, structured_output=structured_output
        )

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
        return _get_openrouter_client(
            model_name, max_tokens=max_tokens, structured_output=structured_output
        )

    raise ValueError(
        f"Unknown model: {model_name}. Set OPENROUTER_API_KEY to route arbitrary models via OpenRouter."
    )
