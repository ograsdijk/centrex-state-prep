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
- **No tests, no linter.** Don't offer to "run the test suite". Verify by
  running examples or `benchmarks/` scripts.
- **Never search or edit under `.venv/`.** It is huge and will flood context.
  Scope Glob/Grep to `src/`, `benchmarks/`, `scripts/`, `examples/`.
- **Notebooks in `examples/` are large and contain executed outputs.** Grep them
  rather than reading whole files; use NotebookEdit for targeted cell changes.
- **`simulator.py` is ~1550 lines.** Read the region you need (offsets) instead
  of the whole file. Public entry points are `Simulator.run` (line ~364) and
  `Simulator.run_microwave_scan` (line ~559).
- **Performance claims need measurements.** `PERFORMANCE_BENCHMARK_RESULTS.md`
  and `benchmarks/common.py` already provide the SPA2 setup — reuse it.
- **Physics constants and unit scalings are intentional.** Flag anything that
  looks wrong; don't silently change it.
- Working branch at time of writing is `batching`; `main` is the default branch.
