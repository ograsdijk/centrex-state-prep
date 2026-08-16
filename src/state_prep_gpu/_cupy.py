from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any


def add_cuda_dll_directories(*, prepend_path: bool = False) -> list[str]:
    """Best-effort: add CUDA DLL directories on Windows.

    On Python 3.8+, Windows DLL search does not reliably honor PATH for
    `LoadLibrary` calls. CuPy (and its dependencies) need CUDA DLLs like
    NVRTC to be discoverable via `os.add_dll_directory`.

    This enables a setup where CUDA libraries come from pip packages such as
    `nvidia-cuda-nvrtc-cu12` instead of a full CUDA Toolkit install.

    Directories are found two ways, since neither alone is sufficient: by
    importing the known `nvidia.*` component packages, and by globbing
    `site-packages/nvidia/*/bin` for anything the list misses.

    Parameters
    ----------
    prepend_path:
        Also prepend the directories to `PATH`. Off by default so importing
        CuPy does not mutate the environment. Benchmarks turn it on because
        NVRTC has been observed to need it in addition to the DLL directories.

    Returns
    -------
    The directories that were added, in order.
    """

    if sys.platform != "win32":
        return []

    # Common pip-provided CUDA component packages.
    candidates = [
        "nvidia.cuda_nvrtc",
        "nvidia.cuda_runtime",
        "nvidia.cublas",
        "nvidia.cusolver",
        "nvidia.cusparse",
        "nvidia.curand",
        "nvidia.cufft",
        "nvidia.nvjitlink",
        "nvidia.cuda_cupti",
    ]

    bin_dirs: list[Path] = []
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue

        mod_path = getattr(mod, "__path__", None)
        if not mod_path:
            continue

        try:
            bin_dirs.append(Path(list(mod_path)[0]) / "bin")
        except Exception:
            continue

    # Glob fallback: catches component packages not in the list above.
    for root in (Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",):
        if root.is_dir():
            bin_dirs.extend(sorted(root.glob("*/bin")))

    added: list[str] = []
    seen: set[str] = set()
    for bin_dir in bin_dirs:
        if not bin_dir.is_dir():
            continue
        path = str(bin_dir)
        if path in seen:
            continue
        seen.add(path)
        try:
            os.add_dll_directory(path)
        except Exception:
            # Non-fatal; continue trying other candidates.
            continue
        added.append(path)

    if added and prepend_path:
        os.environ["PATH"] = os.pathsep.join([*added, os.environ.get("PATH", "")])

    return added


def _add_windows_cuda_dll_dirs() -> None:
    """Backwards-compatible alias for `add_cuda_dll_directories`."""
    add_cuda_dll_directories()


def cupy() -> Any:
    """Import and return `cupy`.

    Kept behind a function so importing `state_prep_gpu` works without CuPy.
    """

    _add_windows_cuda_dll_dirs()

    try:
        import cupy as cp  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CuPy is required for `state_prep_gpu` execution. "
            "Install an appropriate CuPy build (e.g. cupy-cuda12x) and try again."
        ) from exc

    return cp
