#!/usr/bin/env -S just --justfile

# -----------------------------------------------------------------------------
# Development convenience tasks
# -----------------------------------------------------------------------------

# Run the test suite
test:all:
    pytest -q

# Lint the codebase (ruff + mypy + black check)
lint:
    ruff .
    black --check .
    mypy --strict .

# -----------------------------------------------------------------------------
# Micro-benchmarks
# -----------------------------------------------------------------------------

# Run the LCT micro-benchmark locally (CPU by default)
bench:lct size=1024 device="cpu" repeat=50:
    python -m bench.bench_lct --size {{size}} --device {{device}} --repeat {{repeat}}

# Run the same benchmark remotely on Modal (GPU by default).  Any additional
# `bench.bench_lct` flags can be forwarded via the `extra` argument, e.g.
#
#     just bench:lct-modal size=4096 repeat=10 extra="--device cuda"
#
bench:lct-modal size=1024 repeat=50 device="cuda" extra="":
    modal run modal/bench_lct.py::bench --args "--size {{size}} --repeat {{repeat}} --device {{device}} {{extra}}"