from __future__ import annotations

"""Modal entrypoint to run comprehensive benchmarks and tests on H100s."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, TypeVar, Union, cast

import modal

T = TypeVar("T", bound=Union[float, str])

app = modal.App("nanogpt-lct")

def download_dependencies():
    import subprocess, os
    # Ensure that `uv` installs packages into the project-managed environment instead of trying
    # to create nested virtualenvs inside virtualenvs (Modal already provides isolation).

    env = os.environ.copy()
    env["UV_NO_CONFIG"] = "1"  # prevent uv from inspecting the project's pyproject & .venv

    # Delete any pre-existing `.venv` shipped from the host (likely macOS), which contains
    # an incompatible interpreter and confuses uv inside the Linux container.
    import shutil
    host_venv = Path(".venv")
    if host_venv.exists():
        shutil.rmtree(host_venv)

    # Ensure Python 3.12 (matching the project's requirement) is available inside the container.
    subprocess.run(["uv", "python", "install", "3.12"], check=True, env=env)

    # Locate the freshly installed Python 3.12 interpreter. `uv python find` may exit with non-zero if
    # multiple matches exist, so we fall back to globbing the uv python directory.
    try:
        python312 = subprocess.check_output(["uv", "python", "find", "3.12"], env=env).decode().strip()
    except subprocess.CalledProcessError:
        uv_dir = Path(subprocess.check_output(["uv", "python", "dir"], env=env).decode().strip())
        candidates = list(uv_dir.glob("cpython-3.12*/bin/python3.12"))
        if not candidates:
            raise RuntimeError("Python 3.12 not found after installation via uv")
        python312 = str(candidates[0])

    venv_dir = Path("/root/venv")
    if not venv_dir.exists():
        subprocess.run([python312, "-m", "venv", str(venv_dir)], check=True)

    pip_exe = venv_dir / "bin" / "pip"

    # Upgrade pip and install dependencies from pre-generated requirements.txt
    subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
    subprocess.run([
        str(pip_exe),
        "install",
        "--extra-index-url",
        "https://download.pytorch.org/whl/nightly/cu126",
        "--extra-index-url",
        "https://download.pytorch.org/whl/torch_dev.html",
        "--pre",
        "-r",
        "requirements.txt",
    ], check=True)

    # Install the current project in editable mode (so that training scripts can `import modded_nanogpt`)
    subprocess.run([str(pip_exe), "install", "-e", "."], check=True)

    # Make the venv's site-packages visible to the running interpreter so that subsequent imports work.
    import site, sys
    site_packages = venv_dir / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    site.addsitedir(str(site_packages))

image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime")
    # Basic build toolchain and git for editable installs
    .apt_install("git", "curl", "build-essential")
    # Install the standalone `uv` binary (preferred over PyPI wheel for speed).
    # Official installer lives at /install.sh (see https://docs.astral.sh/uv).
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .pip_install("uv")
    .add_local_dir(
        ".", 
        "/root/app",
        ignore=[
            ".git",
            ".git/*",
            ".git/**",
            "__pycache__",
            "**/__pycache__",
            ".pytest_cache",
            ".venv",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".Python",
            "build",
            "develop-eggs",
            "dist",
            "downloads",
            "eggs",
            "lib",
            "lib64",
            "parts",
            "sdist",
            "var",
            "wheels",
            "share/python-wheels",
            "*.egg-info",
            ".installed.cfg",
            "*.egg",
            "MANIFEST",
            ".env",
            ".venv",
            "env/",
            "venv/",
            "ENV/",
            ".env.bak",
            ".venv.bak",
            ".DS_Store"
        ]
    )
)

@app.function(
    image=image,
    gpu="H100",
    memory=32768,
    timeout=3600
)
def run_benchmark(use_lct: bool = False) -> Dict[str, Union[float, str]]:
    """Run a single benchmark with or without LCT."""
    try:
        os.chdir("/root/app")
        download_dependencies()
        
        import torch  # defer import until after potential install
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[modal] Running on device: {device}", flush=True)
        
        start_time = time.time()
        # TODO: Run actual benchmark here
        time.sleep(2)  # Placeholder
        duration = time.time() - start_time
        
        results: Dict[str, Union[float, str]] = {
            "tokens_per_sec": 1000.0,  # Placeholder
            "total_duration": duration,
            "device": device,
            "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else "unknown",
            "torch_version": torch.__version__,
        }
        return results
    except Exception as e:
        print(f"[modal] Benchmark failed: {str(e)}", file=sys.stderr, flush=True)
        raise

@app.function()
async def main():
    """Run both LCT and baseline benchmarks in parallel."""
    try:
        print("[modal] Starting benchmarks...", flush=True)
        
        # Ensure the current container has the same Python environment as the
        # worker containers running `run_benchmark` so that deserialisation of
        # their results (which may reference `torch` objects) succeeds.
        download_dependencies()

        import torch  # noqa: F401 – required for Modal deserialisation
 
        f1 = run_benchmark.remote(use_lct=True)
        f2 = run_benchmark.remote(use_lct=False)
        results = {
            "lct": await f1,
            "baseline": await f2
        }
        
        # Save results to records directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        record_dir = Path("records") / f"{timestamp}_modal_benchmark"
        record_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = record_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"[modal] Results saved to {results_file}", flush=True)
        return results
        
    except Exception as e:
        print(f"[modal] Main failed: {str(e)}", file=sys.stderr, flush=True)
        raise

if __name__ == "__main__":
    app.run() 