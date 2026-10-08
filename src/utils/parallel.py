"""Concurrent LLM requests for independent items (sentences or batches).

Parallelism is an execution setting, not an experimental hyperparameter: it only
changes the order in which independent requests are sent, never their content.
The level comes from ``NORMA_PARALLEL_REQUESTS`` (default 1 = sequential).
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def default_parallel_requests() -> int:
    return max(1, int(os.environ.get("NORMA_PARALLEL_REQUESTS", "1")))


def parallel_map(fn: Callable[[T], R], items: Sequence[T], max_workers: int) -> List[R]:
    """Apply ``fn`` to every item, preserving order; exceptions propagate."""
    if max_workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        return list(executor.map(fn, items))
