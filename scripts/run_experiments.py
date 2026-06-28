#!/usr/bin/env python3
"""
CUDA Boids Automated Experiment Runner — 3 Focused Experiments

  Experiment 1: Algorithm Complexity & Scalability (vary N)
      -> also provides kernel-level data for Experiment 2 (no separate run needed)
  Experiment 3: Hardware Execution Configuration (vary Block Size)
"""

import subprocess
import os
import sys
import time
from pathlib import Path
import json
import datetime
import shutil

# ================
# Configuration
# ================

EXE_PATH = "../build/bin/Release/cis5650_boids_param.exe"
RESULTS_DIR = "../results"

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_DIR, f"run_{TIMESTAMP}")

ALL_METHODS = ["naive", "scattered", "coherent"]

# Experiment 1: Vary N (provides data for both Fig 1 and Fig 2)
EXPERIMENT_1_CONFIG = {
    "name": "vary_n",
    "n_values": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000],
    "methods": ALL_METHODS,
    "block_size": 128,
    "frames": 500,
    "visualize": False,
}

# Experiment 3: Vary Block Size
EXPERIMENT_3_CONFIG = {
    "name": "vary_blocksize",
    "n": 10000,
    "methods": ALL_METHODS,
    "block_sizes": [32, 64, 128, 256, 512, 1024],
    "frames": 500,
    "visualize": False,
}


# ================
# Helpers
# ================

def ensure_results_dir():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(RUN_DIR).mkdir(parents=True, exist_ok=True)


def run_simulation(n, block_size, method, visualize, frames, output_file, timeout=120):
    """Run a single simulation. Returns (success, runtime_seconds)."""
    cmd = [
        EXE_PATH,
        "-n", str(n),
        "-b", str(block_size),
        "-m", method,
        "-f", str(frames),
        "-o", output_file,
        "--no-vis" if not visualize else "-v",
    ]

    vis_label = "vis" if visualize else "novis"
    print(f"\n{'='*72}")
    print(f"  Method={method:10s}  N={n:7d}  BlockSize={block_size:4d}  {vis_label}")
    print(f"  Output: {output_file}")
    print(f"{'='*72}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        runtime = time.time() - start_time
        if result.returncode == 0:
            print(f"  [SUCCESS]  ({runtime:.1f}s)")
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-4:]:
                    print(f"    {line}")
            return True, runtime
        else:
            print(f"  [FAILED]  (code={result.returncode}, {runtime:.1f}s)")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-5:]:
                    print(f"    ERR: {line}")
            return False, runtime
    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        print(f"  [TIMEOUT] after {runtime:.1f}s")
        return False, runtime
    except Exception as e:
        runtime = time.time() - start_time
        print(f"  [ERROR]: {e}")
        return False, runtime


# ================
# Experiments
# ================

def experiment_1_vary_n():
    """Experiment 1: Vary N for all three methods."""
    print("\n" + "="*72)
    print("EXPERIMENT 1: Varying N (Boid Count)")
    print("  Also provides kernel breakdown data for Experiment 2")
    print("="*72)

    config = EXPERIMENT_1_CONFIG
    results = []

    for method in config["methods"]:
        for n in config["n_values"]:
            output_file = os.path.join(
                RUN_DIR,
                f"exp1_{method}_N{n}_B{config['block_size']}_novis.csv"
            )

            # Scale frames to keep runtime manageable
            frames = config["frames"]
            if method == "naive":
                if n >= 50000:
                    frames = 60
                elif n >= 10000:
                    frames = 150
                elif n >= 5000:
                    frames = 250
            else:
                if n >= 500000:
                    frames = 150
                elif n >= 100000:
                    frames = 250

            success, runtime = run_simulation(
                n=n, block_size=config["block_size"], method=method,
                visualize=config["visualize"], frames=frames,
                output_file=output_file
            )

            results.append({
                "experiment": "vary_n",
                "method": method, "n": n,
                "block_size": config["block_size"],
                "frames": frames, "success": success,
                "runtime_seconds": runtime,
                "output_file": output_file
            })
            time.sleep(1)

    return results


def experiment_3_vary_blocksize():
    """Experiment 3: Vary block size for all three methods."""
    print("\n" + "="*72)
    print("EXPERIMENT 3: Varying Block Size")
    print("="*72)

    config = EXPERIMENT_3_CONFIG
    results = []

    for method in config["methods"]:
        for block_size in config["block_sizes"]:
            output_file = os.path.join(
                RUN_DIR,
                f"exp3_{method}_N{config['n']}_B{block_size}_novis.csv"
            )

            success, runtime = run_simulation(
                n=config["n"], block_size=block_size, method=method,
                visualize=config["visualize"], frames=config["frames"],
                output_file=output_file
            )

            results.append({
                "experiment": "vary_blocksize",
                "method": method, "n": config["n"],
                "block_size": block_size,
                "frames": config["frames"], "success": success,
                "runtime_seconds": runtime,
                "output_file": output_file
            })
            time.sleep(1)

    return results


# ================
# Summary
# ================

def save_results_summary(all_results):
    summary_file = os.path.join(RUN_DIR, "experiments_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved -> {summary_file}")
    try:
        shutil.copy(summary_file, os.path.join(RESULTS_DIR, "experiments_summary.json"))
    except Exception:
        pass


def print_summary(all_results):
    print("\n" + "="*72)
    print("EXPERIMENTS SUMMARY")
    print("="*72)

    total = len(all_results)
    ok = sum(1 for r in all_results if r["success"])
    print(f"\nTotal runs: {total}  |  Succeeded: {ok}  |  Failed: {total - ok}")

    if total - ok > 0:
        print("\nFailed runs:")
        for r in all_results:
            if not r["success"]:
                print(f"  FAIL {r['experiment']}: method={r['method']}, N={r['n']}, B={r['block_size']}")

    for exp_name, sort_key in [("vary_n", "n"), ("vary_blocksize", "block_size")]:
        exp = [r for r in all_results if r["experiment"] == exp_name]
        if not exp:
            continue
        exp_label = "Experiment 1 (Vary N)" if exp_name == "vary_n" else "Experiment 3 (Vary Block Size)"
        print(f"\n{exp_label} - by method:")
        for method in ALL_METHODS:
            rows = [r for r in exp if r["method"] == method and r["success"]]
            if rows:
                print(f"\n  {method.upper()}:")
                for r in sorted(rows, key=lambda x: x[sort_key]):
                    print(f"    {sort_key}={r[sort_key]:7d}  {r['runtime_seconds']:6.1f}s  -> {r['output_file']}")



def main():
    print("="*72)
    print("CUDA Boids Performance Experiments")
    print("="*72)

    if not os.path.exists(EXE_PATH):
        print(f"\nERROR: Executable not found at: {EXE_PATH}")
        print("Please build the project in Release mode first.")
        sys.exit(1)

    ensure_results_dir()

    n_exp1 = len(EXPERIMENT_1_CONFIG["methods"]) * len(EXPERIMENT_1_CONFIG["n_values"])
    n_exp3 = len(EXPERIMENT_3_CONFIG["methods"]) * len(EXPERIMENT_3_CONFIG["block_sizes"])
    total = n_exp1 + n_exp3
    print(f"\nExperiment 1 (Vary N):          {n_exp1} runs")
    print(f"  N values:    {EXPERIMENT_1_CONFIG['n_values']}")
    print(f"  Block Size:  {EXPERIMENT_1_CONFIG['block_size']}")
    print(f"\nExperiment 3 (Vary Block Size): {n_exp3} runs")
    print(f"  N:           {EXPERIMENT_3_CONFIG['n']}")
    print(f"  Block Sizes: {EXPERIMENT_3_CONFIG['block_sizes']}")
    print(f"\nTotal runs: {total}")

    all_results = []
    try:
        all_results.extend(experiment_1_vary_n())
        all_results.extend(experiment_3_vary_blocksize())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")

    if all_results:
        save_results_summary(all_results)

    print("\n" + "="*72)
    print("Experiments complete!")
    print(f"Results saved to subfolder -> {RUN_DIR}")
    print("="*72)

    # Run analysis
    print("\nRunning analysis on the results...")
    try:
        subprocess.run(["python", "analyze_experiments.py", RUN_DIR], check=True)
        artifacts = [
            'fig1_steptime_vs_n.png',
            'fig2_kernel_breakdown.png',
            'fig3_steptime_vs_blocksize.png',
            'analysis_report.txt',
        ]
        for art in artifacts:
            src = os.path.join(RUN_DIR, art)
            dst = os.path.join(RESULTS_DIR, art)
            if os.path.exists(src):
                shutil.copy(src, dst)
        print("Analysis artifacts copied to the root results folder successfully!")
    except Exception as e:
        print(f"Error running analysis: {e}")


if __name__ == '__main__':
    main()
