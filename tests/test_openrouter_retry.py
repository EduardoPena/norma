"""Transient OpenRouter failures are retried with backoff; others fail fast."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from openai import APITimeoutError, AuthenticationError, RateLimitError

from src.llm.client import OpenRouterClient


def status_error(error_class, status_code):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return error_class("provider error", response=response, body=None)


def completion(text):
    message = SimpleNamespace(content=text, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class OpenRouterRetryTests(unittest.TestCase):
    def make_client(self, side_effect):
        client = OpenRouterClient.__new__(OpenRouterClient)
        client.model_name = "qwen/qwen3-32b"
        client.max_tokens = None
        client.temperature = 0.0
        client.supported_parameters = None
        create = MagicMock(side_effect=side_effect)
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return client, create

    @patch("time.sleep")
    def test_retries_rate_limits_and_timeouts(self, sleep):
        client, create = self.make_client(
            [
                status_error(RateLimitError, 429),
                APITimeoutError(request=httpx.Request("POST", "https://x")),
                completion("1. YES"),
            ]
        )
        self.assertEqual(client.complete("prompt"), "1. YES")
        self.assertEqual(create.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5.0, 10.0])

    @patch("time.sleep")
    def test_gives_up_after_max_retries(self, sleep):
        client, create = self.make_client(
            [status_error(RateLimitError, 429)] * 10
        )
        with self.assertRaisesRegex(RuntimeError, "OpenRouter API error"):
            client.complete("prompt")
        self.assertEqual(create.call_count, client.max_transient_retries + 1)

    @patch("time.sleep")
    def test_does_not_retry_auth_errors(self, sleep):
        client, create = self.make_client([status_error(AuthenticationError, 401)])
        with self.assertRaises(RuntimeError):
            client.complete("prompt")
        self.assertEqual(create.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
