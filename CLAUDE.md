# CLAUDE.md

Read **[AGENTS.md](AGENTS.md)** — it is the single source of truth for how to
work in this repository (architecture, units, commands, conventions, safety
rails). This file holds only Claude Code-specific notes.

## Quick orientation

`centrex-state-prep` simulates coherent TlF state preparation for CeNTREX.
`src/state_prep` is the CPU OOP engine; `src/state_prep_gpu` is a flat CuPy
batched path. Python 3.11, `uv`, `src/` layout, venv at `.venv/`.

## Claude-specific notes

- **Shell is PowerShell.** Use `.\.venv\Scripts\python.exe`, not `python`, so
  scripts run in the project venv. The Bash tool is also available for POSIX
  syntax, but paths still contain spaces (`examples/SPA/Experimental
  verification/...`) — quote them.
- **There are tests now** (`python -m pytest`, 31 tests + 1 skipped, ~70 s), but still no
  linter. Run them after touching `src/`. They check internal consistency, so
  they will not catch a change in the underlying `centrex_tlf` Hamiltonian —
  for dependency bumps, capture reference outputs before and diff after.
- **Never search or edit under `.venv/`.** It is huge and will flood context.
  Scope Glob/Grep to `src/`, `benchmarks/`, `scripts/`, `tests/`, `examples/`.
- **Notebooks in `examples/` are large and contain executed outputs.** Grep them
  rather than reading whole files; use NotebookEdit for targeted cell changes.
- **`simulator.py` is ~1700 lines.** Read the region you need (offsets) instead
  of the whole file. Public entry points are `Simulator.run` and
  `Simulator.run_microwave_scan` — grep for `def run` rather than trusting a
  line number, since they move.
- **Performance claims need measurements.** `PERFORMANCE_BENCHMARK_RESULTS.md`
  and `benchmarks/common.py` already provide the SPA2 setup — reuse it.
- **Physics constants and unit scalings are intentional.** Flag anything that
  looks wrong; don't silently change it.
- Working branch at time of writing is `batching`; `main` is the default branch.
