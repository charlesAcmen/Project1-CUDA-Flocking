#!/usr/bin/env python3
"""
Automated Experiment Runner for CUDA Boids
Runs three sets of experiments:
1. Vary N (boid count) for all three methods, with and without visualization
2. Vary block size for all three methods
3. Kernel time breakdown comparison across methods
"""

import subprocess
import os
import sys
import time
from pathlib import Path
import json

# ================
# Configuration
# ================

# Path to the parameterized executable (Release build)
EXE_PATH = "../build/bin/Release/cis5650_boids_param.exe"
RESULTS_DIR = "../results"

ALL_METHODS = ["naive", "scattered", "coherent"]

# Experiment 1: Vary N across all three methods
# Use fewer frames for naive at high N to avoid excessive runtime
EXPERIMENT_1_CONFIG = {
    "name": "vary_n",
    # N values - cover a broad range. Naive is O(N^2) so cap at 50k
    "n_values":         [1000, 2000, 5000, 10000, 20000, 50000, 100000],
    "naive_max_n":      20000,   # naive only runs up to this N
    "methods":          ALL_METHODS,
    "block_size":       128,
    "frames":           500,     # number of frames to average over
    "visualize_options": [True, False]
}

# Experiment 2: Vary block size for all three methods
EXPERIMENT_2_CONFIG = {
    "name": "vary_blocksize",
    "n": 10000,
    "methods": ALL_METHODS,
    "block_sizes": [32, 64, 128, 256, 512, 1024],
    "frames": 500,
    "visualize": False   # Pure performance test - no GL overhead
}

# ================
# Helpers
# ================

def ensure_results_dir():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def run_simulation(n, block_size, method, visualize, frames, output_file, timeout=600):
    """
    Run a single simulation with specified parameters.
    Returns: (success, runtime_seconds)
    """
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
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        runtime = time.time() - start_time

        if result.returncode == 0:
            print(f"  [SUCCESS]  ({runtime:.1f}s)")
            if result.stdout:
                # Print last few lines of stdout for progress info
                lines = result.stdout.strip().split('\n')
                for line in lines[-4:]:
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
    """
    Experiment 1: Vary number of boids for all three methods.
    Tests with visualization ON and OFF.
    """
    print("\n" + "="*72)
    print("EXPERIMENT 1: Varying N (Boid Count) - All Three Methods")
    print("="*72)

    config = EXPERIMENT_1_CONFIG
    results = []

    for method in config["methods"]:
        for visualize in config["visualize_options"]:
            vis_str = "vis" if visualize else "novis"

            for n in config["n_values"]:
                # Skip large N for naive (would take too long)
                if method == "naive" and n > config["naive_max_n"]:
                    print(f"\n  [SKIP] naive N={n} > naive_max_n={config['naive_max_n']}")
                    continue

                output_file = os.path.join(
                    RESULTS_DIR,
                    f"exp1_{method}_N{n}_B{config['block_size']}_{vis_str}.csv"
                )

                # Reduce frames for large N to keep runtime reasonable
                frames = config["frames"]
                if method == "naive" and n >= 10000:
                    frames = 200
                elif method == "naive" and n >= 5000:
                    frames = 300

                success, runtime = run_simulation(
                    n=n,
                    block_size=config["block_size"],
                    method=method,
                    visualize=visualize,
                    frames=frames,
                    output_file=output_file
                )

                results.append({
                    "experiment": "vary_n",
                    "method": method,
                    "n": n,
                    "block_size": config["block_size"],
                    "visualize": visualize,
                    "frames": frames,
                    "success": success,
                    "runtime_seconds": runtime,
                    "output_file": output_file
                })

                time.sleep(1)  # Brief cool-down between runs

    return results


def experiment_2_vary_blocksize():
    """
    Experiment 2: Vary block size for all three methods.
    No visualization for pure kernel performance measurement.
    """
    print("\n" + "="*72)
    print("EXPERIMENT 2: Varying Block Size - All Three Methods")
    print("="*72)

    config = EXPERIMENT_2_CONFIG
    results = []

    for method in config["methods"]:
        for block_size in config["block_sizes"]:
            output_file = os.path.join(
                RESULTS_DIR,
                f"exp2_{method}_N{config['n']}_B{block_size}_novis.csv"
            )

            success, runtime = run_simulation(
                n=config["n"],
                block_size=block_size,
                method=method,
                visualize=config["visualize"],
                frames=config["frames"],
                output_file=output_file
            )

            results.append({
                "experiment": "vary_blocksize",
                "method": method,
                "n": config["n"],
                "block_size": block_size,
                "visualize": config["visualize"],
                "frames": config["frames"],
                "success": success,
                "runtime_seconds": runtime,
                "output_file": output_file
            })

            time.sleep(1)

    return results


# ================
# Summary
# ================

def save_results_summary(all_results):
    summary_file = os.path.join(RESULTS_DIR, "experiments_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved → {summary_file}")


def print_summary(all_results):
    print("\n" + "="*72)
    print("EXPERIMENTS SUMMARY")
    print("="*72)

    total = len(all_results)
    successful = sum(1 for r in all_results if r["success"])
    print(f"\nTotal runs: {total}  |  Succeeded: {successful}  |  Failed: {total - successful}")

    if total - successful > 0:
        print("\nFailed runs:")
        for r in all_results:
            if not r["success"]:
                print(f"  FAIL {r['experiment']}: method={r['method']}, N={r['n']}, B={r['block_size']}")

    # Exp1 summary table
    exp1 = [r for r in all_results if r["experiment"] == "vary_n"]
    if exp1:
        print("\nExperiment 1 (Vary N) - by method:")
        for method in ALL_METHODS:
            rows = [r for r in exp1 if r["method"] == method and not r["visualize"] and r["success"]]
            if rows:
                print(f"\n  {method.upper()} (no-vis):")
                for r in sorted(rows, key=lambda x: x["n"]):
                    print(f"    N={r['n']:7d}  {r['runtime_seconds']:6.1f}s  -> {r['output_file']}")

    # Exp2 summary table
    exp2 = [r for r in all_results if r["experiment"] == "vary_blocksize"]
    if exp2:
        print("\nExperiment 2 (Vary Block Size) - by method:")
        for method in ALL_METHODS:
            rows = [r for r in exp2 if r["method"] == method and r["success"]]
            if rows:
                print(f"\n  {method.upper()}:")
                for r in sorted(rows, key=lambda x: x["block_size"]):
                    print(f"    B={r['block_size']:4d}  {r['runtime_seconds']:6.1f}s  -> {r['output_file']}")


def estimate_time(config1, config2):
    """Rough estimate of total experiment time in minutes."""
    runs_exp1 = sum(
        1 for m in config1["methods"]
        for vis in config1["visualize_options"]
        for n in config1["n_values"]
        if not (m == "naive" and n > config1["naive_max_n"])
    )
    runs_exp2 = len(config2["methods"]) * len(config2["block_sizes"])
    total = runs_exp1 + runs_exp2
    # ~500 frames naive@10k ≈ 30s, grid methods much faster
    return total, total * 0.5, total * 2.0


def main():
    print("="*72)
    print("CUDA Boids Performance Experiments - All Three Methods")
    print("="*72)

    # Check executable
    if not os.path.exists(EXE_PATH):
        print(f"\nERROR: Executable not found at: {EXE_PATH}")
        print("Please build the project in Release mode first:")
        print("  cmake --build build --config Release")
        sys.exit(1)

    ensure_results_dir()

    # Print experiment summary
    total_runs, est_min, est_max = estimate_time(EXPERIMENT_1_CONFIG, EXPERIMENT_2_CONFIG)
    print(f"\nExperiment 1 (Vary N):          methods={EXPERIMENT_1_CONFIG['methods']}")
    print(f"  N values:  {EXPERIMENT_1_CONFIG['n_values']}")
    print(f"  naive cap: N <= {EXPERIMENT_1_CONFIG['naive_max_n']}")
    print(f"  vis modes: {EXPERIMENT_1_CONFIG['visualize_options']}")
    print(f"\nExperiment 2 (Vary Block Size): methods={EXPERIMENT_2_CONFIG['methods']}")
    print(f"  N={EXPERIMENT_2_CONFIG['n']}, block sizes={EXPERIMENT_2_CONFIG['block_sizes']}")
    print(f"\nTotal estimated runs: ~{total_runs}")
    print(f"Estimated time:       {est_min:.0f}-{est_max:.0f} minutes\n")

    # Skip interactive prompt for automation
    # response = input("Continue? (y/n): ").strip().lower()
    # if response != 'y':
    #     print("Cancelled.")
    #     sys.exit(0)

    all_results = []
    try:
        all_results.extend(experiment_1_vary_n())
        all_results.extend(experiment_2_vary_blocksize())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")

    if all_results:
        save_results_summary(all_results)
        print_summary(all_results)

    print("\n" + "="*72)
    print("Experiments complete!")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("="*72)
    print("\nNext steps:")
    print("  python analyze_experiments.py")


if __name__ == '__main__':
    main()
