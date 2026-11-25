#!/usr/bin/env python3
"""Nathan's architecture benchmark, with repeated timing and inference memory.

No training data or checkpoints are read. Each architecture uses a fresh process.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

MODELS = ["resnet18", "mobilenetv3_small_100", "deit_tiny_patch16_224"]


def hardware(torch, timm):
    cpu_name = platform.processor()
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    quota = Path("/sys/fs/cgroup/cpu.max")
    return {
        "platform": platform.platform(), "python": sys.version,
        "cpu_name": cpu_name, "logical_cpu_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": quota.read_text().strip() if quota.exists() else None,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch": str(torch.__version__), "timm": timm.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 2**20
        if torch.cuda.is_available() else None,
    }


def timed_forward(model, x, torch, args, cuda=False):
    samples = []
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        if cuda:
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        else:
            baseline = None
        for _ in range(args.trials):
            if cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.runs):
                model(x)
            if cuda:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) / args.runs * 1000)
    result = {"mean_ms": statistics.mean(samples),
              "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
              "trial_mean_ms": samples}
    if cuda:
        result.update(
            resident_mb=baseline / 2**20,
            peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
            incremental_peak_mb=(torch.cuda.max_memory_allocated() - baseline) / 2**20,
            peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20,
        )
    return result


def worker(args):
    import torch
    import timm
    from torch.utils.flop_counter import FlopCounterMode

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: select a GPU runtime before running this benchmark.")
    model = timm.create_model(args.worker, pretrained=False, num_classes=14).eval()
    x = torch.randn(1, 3, 224, 224)
    params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        assert tuple(model(x).shape) == (1, 14)
        # Profile explicit attention matmuls because the fused CPU SDPA path can
        # be absent from FlopCounterMode's registered formulas. Restore the
        # default fused path before latency/memory measurement.
        attention = [(m, m.fused_attn) for m in model.modules() if hasattr(m, "fused_attn")]
        for module, _ in attention:
            module.fused_attn = False
        try:
            with FlopCounterMode(display=False) as counter:
                model(x)
        finally:
            for module, enabled in attention:
                module.fused_attn = enabled
    flops = int(counter.get_total_flops())
    assert flops > 0
    breakdown = {str(k): int(v) for k, v in counter.get_flop_counts().get("Global", {}).items()}
    cpu = timed_forward(model, x, torch, args)
    del model, x
    gc.collect()
    gpu = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        model = timm.create_model(args.worker, pretrained=False, num_classes=14).eval().cuda()
        x = torch.randn(1, 3, 224, 224, device="cuda")
        gpu = timed_forward(model, x, torch, args, cuda=True)
        del model, x
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "model": args.worker, "parameter_count": params, "params_m": params / 1e6,
        "forward_flops": flops, "gflops": flops / 1e9, "gmacs": flops / 2e9,
        "cpu_latency_ms": cpu["mean_ms"], "cpu_latency_std_ms": cpu["std_ms"],
        "gpu_latency_ms": gpu["mean_ms"] if gpu else None,
        "gpu_latency_std_ms": gpu["std_ms"] if gpu else None,
        "gpu_peak_allocated_mb": gpu["peak_allocated_mb"] if gpu else None,
        "gpu_resident_mb": gpu["resident_mb"] if gpu else None,
        "gpu_incremental_peak_mb": gpu["incremental_peak_mb"] if gpu else None,
        "batch_size": 1, "input_shape": "1x3x224x224", "num_classes": 14,
        "dtype": "float32", "cpu_threads": args.threads,
        "warmup_iters": args.warmup, "timed_iters_per_trial": args.runs,
        "trials": args.trials,
    }
    details = {"summary": result, "cpu_timing": cpu, "gpu_timing": gpu,
               "counted_flop_operators": breakdown, "flop_profile_attention": "explicit matmul; default fused path restored for timing", "hardware": hardware(torch, timm)}
    (args.output_dir / f"{args.worker}.json").write_text(json.dumps(details, indent=2))
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--worker", choices=MODELS)
    args = p.parse_args()
    if min(args.threads, args.warmup, args.runs, args.trials) < 1:
        p.error("threads, warmup, runs, and trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    results = []
    for name in MODELS:
        print(f"Benchmarking {name}: CPU and GPU, batch size 1", flush=True)
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", name,
               "--output-dir", str(args.output_dir.resolve()), "--threads", str(args.threads),
               "--warmup", str(args.warmup), "--runs", str(args.runs), "--trials", str(args.trials)]
        if args.require_cuda:
            cmd.append("--require-cuda")
        subprocess.run(cmd, check=True)
        details = json.loads((args.output_dir / f"{name}.json").read_text())
        results.append(details["summary"])
        with (args.output_dir / "efficiency_benchmark.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    (args.output_dir / "benchmark_protocol.json").write_text(json.dumps({
        "models": MODELS, "initialization": "random; pretrained=False; no checkpoint",
        "scope": "architecture-level, model-forward-only inference; preprocessing and data transfer excluded",
        "execution": "timm eager eval, torch.no_grad, FP32, TF32 disabled, cudnn.benchmark=False",
        "input_shape": [1, 3, 224, 224], "num_classes": 14,
        "cpu_threads": args.threads, "warmup_iters": args.warmup,
        "timed_iters_per_trial": args.runs, "trials": args.trials,
        "latency_statistic": "mean of trial means; standard deviation across trial means",
        "gpu_timing": "perf_counter; CUDA synchronization before and after each timed block",
        "complexity": "PyTorch FlopCounterMode supported-operator estimates; GFLOPs=FLOPs/1e9; GMACs=FLOPs/2e9",
        "memory": "PyTorch peak allocated MiB after warmup and peak reset; includes model/input/inference tensors; excludes CUDA context and non-PyTorch allocations",
        "isolation": "fresh subprocess per architecture; CPU model deleted before GPU model allocation",
    }, indent=2))
    print("Benchmark complete:", args.output_dir / "efficiency_benchmark.csv", flush=True)


if __name__ == "__main__":
    main()
