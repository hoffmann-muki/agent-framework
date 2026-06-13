# Copyright (c) Microsoft. All rights reserved.

"""Optional NVIDIA NVTX range helpers.

The helper is intentionally dependency-free unless ``MAF_NVTX_ENABLE`` is set.
When enabled, it uses the optional ``nvtx`` package if installed and otherwise
falls back to no-op ranges.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

_ENABLED = os.getenv("MAF_NVTX_ENABLE", "").lower() in {"1", "true", "yes"}
_nvtx = None

if _ENABLED:
    try:
        import nvtx as _nvtx  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - optional profiler dependency
        _nvtx = None


@contextlib.contextmanager
def range_push(name: str) -> Iterator[None]:
    """Push an NVTX range when optional NVTX instrumentation is enabled."""
    if _nvtx is None:
        yield
        return

    with _nvtx.annotate(message=name):  # type: ignore[union-attr]
        yield
