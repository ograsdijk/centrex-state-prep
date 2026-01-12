from __future__ import annotations

from typing import Any


def cupy() -> Any:
    """Import and return `cupy`.

    Kept behind a function so importing `state_prep_gpu` works without CuPy.
    """

    try:
        import cupy as cp  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CuPy is required for `state_prep_gpu` execution. "
            "Install an appropriate CuPy build (e.g. cupy-cuda12x) and try again."
        ) from exc

    return cp
