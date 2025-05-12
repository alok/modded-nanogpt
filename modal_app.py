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
    # Ensure CUDA-enabled torch is present; install silently if missing
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--quiet",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
                "torch==2.2.1+cu121",
            ],
            check=True,
        )
    # Install project in editable mode (dependencies declared in pyproject.toml)
    subprocess.run(["uv", "pip", "install", "-e", "."], check=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime")
    # Basic build toolchain and git for editable installs
    .apt_install("git", "curl", "build-essential")
    # Install uv (Rust binary) and ensure it's first on PATH
    .run_commands("curl -Ls https://astral.sh/uv/install | bash")
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