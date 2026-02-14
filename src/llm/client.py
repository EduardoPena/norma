"""LLM client factory and implementations"""
from abc import ABC, abstractmethod
from typing import Optional
import os


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
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e


class MaritacaAIClient(LLMClient):
    """Maritaca client"""
    
    def __init__(self, api_key: str, model_name: str = "sabiazinho-4"):
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key,  base_url="https://chat.maritaca.ai/api")
        self.model_name = model_name
        # GPT-5 and some models only support temperature=1
        self.temperature = 1 if model_name.startswith("sabia") else 0.0
    
    def complete(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
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
                messages=[{"role": "user", "content": prompt}]
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
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Mistral API error: {e}") from e


class GroqClient(LLMClient):
    """Groq client (Llama, Mixtral, Gemma, Qwen, etc.)"""
    
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model_name = model_name
    
    def complete(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}") from e


def get_llm_client(model_name: str = "gpt-5-mini") -> LLMClient:
    """
    Factory to create LLM clients based on model name
    
    Args:
        model_name: Model identifier (e.g., "gpt-5-mini", "claude-3-5-sonnet-20241022")
        
    Returns:
        LLMClient instance
        
    Raises:
        ValueError: If API key is missing or model is unknown
    """
    if model_name.startswith("gpt-"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAIClient(api_key=api_key, model_name=model_name)
    
    if model_name.startswith("sabia"):
        api_key = os.getenv("MARITACA_API_KEY")
        if not api_key:
            raise ValueError("MARITACA_API_KEY environment variable not set")
        return MaritacaAIClient(api_key=api_key, model_name=model_name)
    
    if model_name.startswith("claude-"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return AnthropicClient(api_key=api_key, model_name=model_name)
    
    if model_name.startswith("mistral-"):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        return MistralClient(api_key=api_key, model_name=model_name)
    
    if model_name.startswith(("llama-", "moonshotai", "qwen", "gemma", "mixtral")):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        return GroqClient(api_key=api_key, model_name=model_name)
    
    raise ValueError(f"Unknown model: {model_name}")