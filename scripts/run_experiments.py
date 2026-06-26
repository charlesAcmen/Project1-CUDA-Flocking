#!/usr/bin/env python3
"""
Automated Experiment Runner for CUDA Boids
Runs two sets of experiments:
1. Vary N (boid count) with visualization ON/OFF
2. Vary block size to find optimal performance
"""

import subprocess
import os
import sys
import time
from pathlib import Path
import json

# Configuration
EXE_PATH = "../build/bin/Release/cis5650_boids.exe"
RESULTS_DIR = "../results"

# Experiment 1: Vary N
EXPERIMENT_1_CONFIG = {
    "name": "vary_n",
    "n_values": [1000, 2000, 5000, 10000, 20000],
    "methods": ["naive"],  # Start with naive, add others when implemented
    "block_size": 128,
    "frames": 500,
    "visualize_options": [True, False]
}

# Experiment 2: Vary block size
EXPERIMENT_2_CONFIG = {
    "name": "vary_blocksize",
    "n": 10000,
    "methods": ["naive"],  # Will test all methods when implemented
    "block_sizes": [32, 64, 128, 256, 512, 1024],
    "frames": 500,
    "visualize": False  # Pure performance test
}

def ensure_results_dir():
    """Create results directory if it doesn't exist"""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

def run_simulation(n, block_size, method, visualize, frames, output_file):
    """
    Run a single simulation with specified parameters
    Returns: (success, runtime_seconds)
    """
    method_str = method if isinstance(method, str) else str(method)
    
    cmd = [
        EXE_PATH,
        "-n", str(n),
        "-b", str(block_size),
        "-m", method_str,
        "-f", str(frames),
        "-o", output_file
    ]
    
    if visualize:
        cmd.append("-v")
    else:
        cmd.append("--no-vis")
    
    print(f"\n{'='*70}")
    print(f"Running: N={n}, BlockSize={block_size}, Method={method_str}, Vis={visualize}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            timeout=300,  # 5 minute timeout
            capture_output=True,
            text=True
        )
        
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ SUCCESS - Runtime: {runtime:.2f}s")
            return True, runtime
        else:
            print(f"✗ FAILED - Return code: {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            return False, runtime
            
    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        print(f"✗ TIMEOUT after {runtime:.2f}s")
        return False, runtime
    except Exception as e:
        runtime = time.time() - start_time
        print(f"✗ ERROR: {e}")
        return False, runtime

def experiment_1_vary_n():
    """
    Experiment 1: Vary number of boids
    Test with visualization ON and OFF to compare performance impact
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Varying N (Boid Count)")
    print("="*70)
    
    config = EXPERIMENT_1_CONFIG
    results = []
    
    for method in config["methods"]:
        for visualize in config["visualize_options"]:
            vis_str = "vis" if visualize else "novis"
            
            for n in config["n_values"]:
                output_file = os.path.join(
                    RESULTS_DIR,
                    f"exp1_{method}_N{n}_B{config['block_size']}_{vis_str}.csv"
                )
                
                success, runtime = run_simulation(
                    n=n,
                    block_size=config["block_size"],
                    method=method,
                    visualize=visualize,
                    frames=config["frames"],
                    output_file=output_file
                )
                
                results.append({
                    "experiment": "vary_n",
                    "method": method,
                    "n": n,
                    "block_size": config["block_size"],
                    "visualize": visualize,
                    "frames": config["frames"],
                    "success": success,
                    "runtime_seconds": runtime,
                    "output_file": output_file
                })
                
                # Small delay between runs
                time.sleep(2)
    
    return results

def experiment_2_vary_blocksize():
    """
    Experiment 2: Vary block size to find optimal performance
    Test without visualization for pure kernel performance
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Varying Block Size")
    print("="*70)
    
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
            
            # Small delay between runs
            time.sleep(2)
    
    return results

def save_results_summary(all_results):
    """Save a JSON summary of all experiment results"""
    summary_file = os.path.join(RESULTS_DIR, "experiments_summary.json")
    
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results summary saved to: {summary_file}")
    print(f"{'='*70}")

def print_summary(all_results):
    """Print a summary of all experiments"""
    print("\n" + "="*70)
    print("EXPERIMENTS SUMMARY")
    print("="*70)
    
    total = len(all_results)
    successful = sum(1 for r in all_results if r["success"])
    failed = total - successful
    
    print(f"\nTotal runs: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed runs:")
        for r in all_results:
            if not r["success"]:
                print(f"  - {r['experiment']}: N={r['n']}, B={r['block_size']}, "
                      f"Method={r['method']}, Vis={r['visualize']}")
    
    print("\nExperiment 1 (Vary N):")
    exp1_results = [r for r in all_results if r["experiment"] == "vary_n"]
    if exp1_results:
        for vis in [True, False]:
            vis_results = [r for r in exp1_results if r["visualize"] == vis]
            if vis_results:
                vis_str = "WITH visualization" if vis else "WITHOUT visualization"
                print(f"\n  {vis_str}:")
                for r in vis_results:
                    status = "✓" if r["success"] else "✗"
                    print(f"    {status} N={r['n']:5d} - {r['runtime_seconds']:.2f}s")
    
    print("\nExperiment 2 (Vary Block Size):")
    exp2_results = [r for r in all_results if r["experiment"] == "vary_blocksize"]
    if exp2_results:
        for r in exp2_results:
            status = "✓" if r["success"] else "✗"
            print(f"  {status} BlockSize={r['block_size']:4d} - {r['runtime_seconds']:.2f}s")

def main():
    """Run all experiments"""
    print("="*70)
    print("CUDA Boids Performance Experiments")
    print("="*70)
    
    # Check if executable exists
    if not os.path.exists(EXE_PATH):
        print(f"\nError: Executable not found at {EXE_PATH}")
        print("Please compile the project first.")
        sys.exit(1)
    
    # Create results directory
    ensure_results_dir()
    
    # Confirm before running
    print("\nThis will run the following experiments:")
    print(f"1. Vary N: {len(EXPERIMENT_1_CONFIG['n_values'])} values × "
          f"{len(EXPERIMENT_1_CONFIG['visualize_options'])} vis options × "
          f"{len(EXPERIMENT_1_CONFIG['methods'])} methods = "
          f"{len(EXPERIMENT_1_CONFIG['n_values']) * len(EXPERIMENT_1_CONFIG['visualize_options']) * len(EXPERIMENT_1_CONFIG['methods'])} runs")
    print(f"2. Vary Block Size: {len(EXPERIMENT_2_CONFIG['block_sizes'])} values × "
          f"{len(EXPERIMENT_2_CONFIG['methods'])} methods = "
          f"{len(EXPERIMENT_2_CONFIG['block_sizes']) * len(EXPERIMENT_2_CONFIG['methods'])} runs")
    
    total_runs = (len(EXPERIMENT_1_CONFIG['n_values']) * 
                  len(EXPERIMENT_1_CONFIG['visualize_options']) * 
                  len(EXPERIMENT_1_CONFIG['methods']) +
                  len(EXPERIMENT_2_CONFIG['block_sizes']) * 
                  len(EXPERIMENT_2_CONFIG['methods']))
    
    print(f"\nTotal: {total_runs} runs")
    print(f"Estimated time: {total_runs * 1.5:.0f}-{total_runs * 3:.0f} minutes\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Run experiments
    all_results = []
    
    try:
        # Experiment 1: Vary N
        exp1_results = experiment_1_vary_n()
        all_results.extend(exp1_results)
        
        # Experiment 2: Vary block size
        exp2_results = experiment_2_vary_blocksize()
        all_results.extend(exp2_results)
        
    except KeyboardInterrupt:
        print("\n\nExperiments interrupted by user!")
    
    # Save and print results
    if all_results:
        save_results_summary(all_results)
        print_summary(all_results)
    
    print("\n" + "="*70)
    print("Experiments complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python analyze_experiments.py")
    print("2. Check generated plots and reports")

if __name__ == '__main__':
    main()
