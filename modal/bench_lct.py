"""modal/bench_lct.py
---------------------------------
Run the LCT micro-benchmark inside a Modal GPU container so you can execute it
from **any** host (macOS included) without worrying about native `uv` wheels or
CUDA libraries. All heavy lifting happens in a Linux container provisioned by
Modal.

Usage
-----
Locally (with your `.env` holding the Modal token/secret)::

    # Default run (N=1024, device=cuda, 50 repeats)
    modal run modal/bench_lct.py::bench

    # Custom size / repeats
    modal run modal/bench_lct.py::bench --args "--size 4096 --repeat 10"

The repository is mounted read-only into the container at `/workspace`. On each
invocation we

1. `uv pip install -e .[dev]` to get all project dependencies (GPU wheels are
   resolved automatically because the container is Linux/amd64).
2. Execute the regular Python benchmark module (`bench.bench_lct`).

The Image is built **once** and then cached by Modal, so subsequent invocations
incur minimal overhead (<1 s cold-start on typical accounts).
"""

from __future__ import annotations

import modal


# ----------------------------------------------------------------------------
# Build container image
# ----------------------------------------------------------------------------

# Base: lightweight Debian with the requested Python version (must match
# `pyproject.toml` → 3.12).*  We then install system packages required by
# PyTorch wheels (`build-essential` for triton) and the `uv` package manager.
#
# *Modal automatically ensures the correct minor release is available on the
#  backend.  If you need a different version, change here and in `pyproject`.

PYTHON_VERSION = "3.12"

base_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "curl", "build-essential")
    # Install uv (we rely on it to honour the Linux-only wheel map in
    # `pyproject.toml` so that the correct CUDA build of PyTorch is pulled).
    .run_commands(["curl -Ls https://astral.sh/uv/install | bash"])
)


# ----------------------------------------------------------------------------
# Modal stub & function
# ----------------------------------------------------------------------------

stub = modal.Stub("lct-bench")


@stub.function(
    image=base_image,
    gpu="A10G",  # Feel free to change ("any", "T4", "A100", …)
    secrets=[modal.Secret.from_dotenv()],
    mounts=[modal.Mount.from_local_dir(".", remote_path="/workspace", exclude=[
        ".git",
        "records",
        "records-medium",
        "data",
        "**/*.pdf",
        "**/*.png",
        "**/*.eps",
        "*.log",
        "*.aux",
        "*.out",
        "*.fdb_latexmk",
    ])],
    timeout=60 * 30,  # 30 minutes – plenty for iterative runs
)
def bench(*bench_args: str):  # noqa: D401
    """Entrypoint executed in the remote GPU container.

    Parameters
    ----------
    *bench_args
        Optional arguments forwarded verbatim to ``bench.bench_lct``.  For
        example, ``("--size", "4096", "--repeat", "100", "--device",
        "cuda")``.
    """

    import os
    import subprocess
    import sys

    os.chdir("/workspace")

    # ---------------------------------------------------------------------
    # 1. Install project in editable mode (GPU wheels resolved by uv)
    # ---------------------------------------------------------------------
    install_cmd = ["uv", "pip", "install", "-e", ".[dev]"]
    print("[modal] Installing project with:", " ".join(install_cmd), flush=True)
    subprocess.run(install_cmd, check=True)

    # ---------------------------------------------------------------------
    # 2. Run benchmark – default to GPU if present
    # ---------------------------------------------------------------------
    default_args = ("--device", "cuda") if not bench_args else ()
    cmd = [sys.executable, "-m", "bench.bench_lct", *bench_args, *default_args]

    print("[modal] Running benchmark:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
