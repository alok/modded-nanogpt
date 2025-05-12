#!/usr/bin/env -S just --justfile

# -----------------------------------------------------------------------------
# Development convenience tasks
# -----------------------------------------------------------------------------

# Run the test suite
"test:all":
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
"bench:lct": size=1024 device="cpu" repeat=50:
    python -m bench.bench_lct --size {{size}} --device {{device}} --repeat {{repeat}}

# Run the same benchmark remotely on Modal (GPU by default).  Any additional
# `bench.bench_lct` flags can be forwarded via the `extra` argument, e.g.
#
#     just bench:lct-modal size=4096 repeat=10 extra="--device cuda"
#
"bench:lct-modal": size=1024 repeat=50 device="cuda" extra="":
    modal run modal/bench_lct.py::bench --args "--size {{size}} --repeat {{repeat}} --device {{device}} {{extra}}"

# -----------------------------------------------------------------------------
# Modal application helpers
# -----------------------------------------------------------------------------

# Run the Modal main function after ensuring Git's fsmonitor socket is gone.
# This avoids the "Operation not supported on socket" error when Modal tries
# to package the repo (it cannot copy Unix domain sockets).  We first stop the
# daemon, then remove any leftover socket files, and finally invoke the Modal
# run command.  Extra CLI flags can be forwarded via `args`.
"modal:run": args="":
    # Stop git fsmonitor if running – ignore failure if already stopped
    git fsmonitor--daemon stop || true
    # Remove lingering socket files so Modal's file walker doesn't choke
    rm -f .git/fsmonitor--daemon .git/fsmonitor--daemon.ipc || true
    # Execute the Modal job
    modal run modal_app.py::main {{args}}