from __future__ import annotations

from typing import Any


def torch() -> Any:
    """Import and return PyTorch only when a GPU entry point is used."""

    try:
        import torch as torch_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required to run `state_prep_gpu`. Install a CUDA-enabled "
            "PyTorch build for GPU execution, or a CPU build for small checks."
        ) from exc

    return torch_module
