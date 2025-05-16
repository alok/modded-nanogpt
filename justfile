#!/usr/bin/env -S just --justfile

# -----------------------------------------------------------------------------
# Development convenience tasks
# -----------------------------------------------------------------------------

# Run the test suite
test_all:
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
# bench_lct size=1024 device="cpu" repeat=50:
#     python -m bench.bench_lct --size {{size}} --device {{device}} --repeat {{repeat}}

# Run the same benchmark remotely on Modal (GPU by default).  Any additional
# `bench.bench_lct` flags can be forwarded via the `extra` argument, e.g.
#
#     just bench:lct-modal size=4096 repeat=10 extra="--device cuda"
#
# bench_lct_modal size=1024 repeat=50 device="cuda" extra="":
#     modal run modal/bench_lct.py::bench --args "--size {{size}} --repeat {{repeat}} --device {{device}} {{extra}}"

# -----------------------------------------------------------------------------
# Modal application helpers
# -----------------------------------------------------------------------------

# Run the Modal main function after ensuring Git's fsmonitor socket is gone.
# This avoids the "Operation not supported on socket" error when Modal tries
# to package the repo (it cannot copy Unix domain sockets).  We first stop the
# daemon, then remove any leftover socket files, and finally invoke the Modal
# run command.  Extra CLI flags can be forwarded via `args`.
modal_run:
    # Stop git fsmonitor if running – ignore failure if already stopped
    git fsmonitor--daemon stop || true
    # Remove lingering socket files so Modal's file walker doesn't choke
    rm -f .git/fsmonitor--daemon .git/fsmonitor--daemon.ipc || true
    # Execute the Modal job
    modal run modal_app.py::main

# nanogpt commands

# Default command, runs when you just type `just`
default:
    @echo "Available commands:"
    @echo "  just test          - Run all pytests"
    @echo "  just lint          - Run ruff and mypy linters"
    @echo "  just bench-lct     - Run LCT benchmark (baseline vs LCT)"
    @echo "  just paper         - Compile the NeurIPS paper"

# Run all tests
test:
    @pytest -q --cov=torchlayers --cov-report=term-missing

# Lint the codebase
lint:
    @ruff check .
    @mypy --strict .

# Benchmark LCT Layer
# Ensure train_gpt.py is using the appropriate model settings for this benchmark
# The script bench/bench_lct.py saves JSON and prints tokens/sec
bench-lct:
    @echo "Running LCT benchmark (baseline)..."
    @python bench/bench_lct.py --use-lct-in-block false
    @echo "Running LCT benchmark (LCT enabled)..."
    @python bench/bench_lct.py --use-lct-in-block true
    @echo "Benchmark complete. Results in ./records/"

# Compile the NeurIPS paper
paper:build:
    latexmk -pdf -silent -cd paper/main.tex

paper:watch:
    latexmk -pdf -pvc -cd paper/main.tex

paper:clean:
    latexmk -C -cd paper/main.tex

# Install/sync dependencies
install:
    @uv sync

# Bump version (example, actual versioning might be more complex)
# version:bump PATCH_TYPE=\"patch\":
# @poetry version {{PATCH_TYPE}}
# @echo "Version bumped. Remember to commit pyproject.toml and CHANGELOG.md"

.PHONY: default test lint bench-lct paper install