#!/usr/bin/env python3
"""Measure selected static v0.2 artifacts without fitting or cohort access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
import numpy as np
import torch
from modelchord import Artifact
from text_features import text_features


def timings(function, warmup: int, repeats: int):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"batch": 1, "warmup": warmup, "repeats": repeats,
            "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "min_ms": min(samples), "max_ms": max(samples),
            "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v02/runtime-profile.json")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior measurements; choose a fresh output")
    if not 1 <= args.warmup <= 100 or not 20 <= args.repeats <= 1000:
        raise ValueError("Bounded warmup/repeat counts required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    lock_path = ROOT / "experiments/v02/confirmation-lock.json"
    lock = json.loads(lock_path.read_text())
    raw_pixels = ((np.arange(64) * 7) % 17).astype(np.float32).reshape(8, 8)
    raw_text = "Meeting moved to 3 pm. Please bring the updated notes."
    rows = {}
    with torch.inference_mode():
        for suite, relative in lock["deployments"].items():
            path = ROOT / relative
            artifact = Artifact.open(path)
            model = artifact.model()
            if suite == "vision":
                preprocess = lambda: torch.tensor((raw_pixels / 16).reshape(1, 64), dtype=torch.float32)
                input_description = "Deterministic synthetic 8x8 integer pixels in [0,16]; flatten/divide by16"
            else:
                preprocess = lambda: torch.tensor(text_features([raw_text]))
                input_description = f"Fixed synthetic {len(raw_text)}-character scheduling sentence; clean signed byte features"
            x = preprocess()

            def postprocess(logits):
                probabilities = logits.softmax(1)
                return artifact.manifest["architecture"]["classes"][int(probabilities.argmax(1)[0])]

            def integrated_request():
                return postprocess(Artifact.open(path).predict(preprocess()))

            def loaded_model_request():
                return postprocess(model(preprocess()))

            functions = {
                "preprocessing_only": preprocess,
                "warmed_forward_logits": lambda: model(x),
                "artifact_predict_cached_handle": lambda: artifact.predict(x),
                "artifact_open_and_predict": lambda: Artifact.open(path).predict(x),
                "warmed_loaded_model_request": loaded_model_request,
                "safe_integrated_request": integrated_request,
            }
            torch.testing.assert_close(model(x), artifact.predict(x), atol=0, rtol=0)
            files = {p.name: p.stat().st_size for p in path.iterdir() if p.is_file()}
            last_layer = f"layers.{len(model.layers) - 1}."
            head_bytes = sum(int(np.prod(meta["shape"])) * 4 for name, meta in artifact.manifest["tensors"].items()
                             if name.startswith(last_layer))
            rows[suite] = {
                "artifact": relative,
                "weights_sha256": artifact.manifest["weights_sha256"],
                "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
                "files_bytes": files, "total_stored_bytes_all_heads_included": sum(files.values()),
                "weights_file_bytes": files["model.safetensors"], "last_classifier_layer_tensor_bytes": head_bytes,
                "all_parameters": sum(p.numel() for p in model.parameters()),
                "input_description": input_description,
                "measurement": {name: timings(function, args.warmup, args.repeats)
                                for name, function in functions.items()},
            }
    hardware_query = subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                    check=False, capture_output=True, text=True)
    hardware = hardware_query.stdout.strip() if hardware_query.returncode == 0 else "unavailable under current permissions"
    record = {
        "schema": "modelchord-v02-runtime-profile/v1", "created_unix": time.time(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "confirmation_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "host": {"hardware": hardware, "hardware_query_returncode": hardware_query.returncode,
                 "hardware_query_stderr": hardware_query.stderr.strip(),
                 "platform": platform.platform(), "machine": platform.machine(),
                 "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                 "device": "CPU", "torch_intraop_threads": torch.get_num_threads(),
                 "torch_interop_threads": torch.get_num_interop_threads()},
        "scope": {
            "warmed_forward_logits": "Already loaded static model, preprocessed batch1; logits only",
            "artifact_predict_cached_handle": "Existing validated Artifact handle; predict rehashes/loads all weights per call",
            "artifact_open_and_predict": "Manifest validation/open plus per-call model load and logits",
            "warmed_loaded_model_request": "Preprocessing plus already loaded forward, softmax and label selection",
            "safe_integrated_request": "Preprocessing plus Artifact.open/predict, safe per-call validation/load, softmax and label selection",
            "exclusions": "No process/import/CLI startup, HTTP transport, training, parent experts or cohort data",
            "cache_state": "All paths warmed; operating-system file cache is warm, not a cold-disk benchmark",
            "sampling": f"{args.repeats} repeats after{args.warmup} warmups; sequential same-process measurements; local host background activity not controlled",
            "rss_scope": "Process high-water RSS includes imports, both artifacts and every timing block",
        },
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "artifacts": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({suite: {"stored_bytes": row["total_stored_bytes_all_heads_included"],
                             **{name: {k: v for k, v in result.items() if k in ["p50_ms", "p95_ms"]}
                                for name, result in row["measurement"].items()}}
                      for suite, row in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
