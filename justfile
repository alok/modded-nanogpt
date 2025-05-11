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

# Micro-benchmark for the LCT layer (CPU by default)
bench:lct size=1024 device="cpu":
    python -m bench.bench_lct --size {{size}} --device {{device}}
