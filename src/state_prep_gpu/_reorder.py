from __future__ import annotations

from typing import Any, Tuple

from ._cupy import cupy


def reorder_evecs_batched(
    V_in: Any,
    E_in: Any,
    V_ref: Any,
) -> Tuple[Any, Any]:
    """Reorder eigenvectors/energies to maximize overlap with a reference.

    Mirrors `state_prep.utils.reorder_evecs` for batched matrices, but **not its
    algorithm**: this matches greedily with `argmax`, while the CPU path solves
    an optimal one-to-one assignment. Greedy matching does not guarantee a
    permutation -- when two input eigenvectors have their largest overlap with
    the same reference column, which happens once states mix strongly, one
    reference column is claimed twice and the subsequent `argsort` produces an
    arbitrary ordering rather than a matching. GPU labels can therefore differ
    from CPU labels for the same trajectory.

    This is left as-is deliberately: there is no CuPy `linear_sum_assignment`, so
    fixing it needs a host round-trip or a batched auction kernel, and the labels
    it produces should not be load-bearing in the first place. Identify states by
    quantum numbers instead -- `state_prep.utils.eigenstate_quantum_numbers` with
    `select_eigenstate`, or `result.population(...)` -- which is immune to
    labelling entirely. See the crossing analysis in `IMPROVEMENTS.md`.

    Parameters
    ----------
    V_in:
        (..., n, n) eigenvectors (columns are eigenvectors).
    E_in:
        (..., n) eigenvalues.
    V_ref:
        (..., n, n) reference eigenvectors.

    Returns
    -------
    E_out, V_out:
        Reordered energies/vectors.
    """

    cp = cupy()

    # overlap[..., i, j] = | <v_in_i | v_ref_j> |
    overlap = cp.abs(cp.matmul(cp.conj(cp.swapaxes(V_in, -2, -1)), V_ref))

    # For each input eigenvector i, find which ref eigenvector matches best.
    best_j = cp.argmax(overlap, axis=-1)

    # Sort input eigenvectors by their matched ref index.
    order = cp.argsort(best_j, axis=-1)

    # Gather along the eigenvector axis (-1) for V_in and E_in.
    V_out = cp.take_along_axis(V_in, order[..., None, :], axis=-1)
    E_out = cp.take_along_axis(E_in, order, axis=-1)

    return E_out, V_out
