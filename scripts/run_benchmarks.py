#!/usr/bin/env python3
"""
Automated Benchmark Script for CUDA Boids Flocking
Runs different configurations and collects performance data
"""

import subprocess
import time
import os
import sys
from pathlib import Path

# Configuration
BOID_COUNTS = [1000, 5000, 10000, 20000, 50000]
METHODS = ['naive', 'scattered', 'coherent']
FRAMES_PER_TEST = 500  # Number of frames to collect per test
EXECUTABLE = '../build/Release/cis5650_flock.exe'  # Adjust path as needed

def compile_with_settings(visualize, method):
    """
    Modify main.cpp with desired settings and recompile
    Note: This is a simple example. You may want to use CMake defines instead.
    """
    print(f"Configuring for {method} method, visualize={visualize}...")
    
    # Read main.cpp
    main_cpp_path = Path('../src/main.cpp')
    with open(main_cpp_path, 'r') as f:
        content = f.read()
    
    # Backup original
    backup_path = main_cpp_path.with_suffix('.cpp.backup')
    if not backup_path.exists():
        with open(backup_path, 'w') as f:
            f.write(content)
    
    # Modify defines
    content = content.replace('#define VISUALIZE 1', f'#define VISUALIZE {1 if visualize else 0}')
    
    if method == 'naive':
        content = content.replace('#define UNIFORM_GRID 1', '#define UNIFORM_GRID 0')
        content = content.replace('#define COHERENT_GRID 1', '#define COHERENT_GRID 0')
    elif method == 'scattered':
        content = content.replace('#define UNIFORM_GRID 0', '#define UNIFORM_GRID 1')
        content = content.replace('#define COHERENT_GRID 1', '#define COHERENT_GRID 0')
    elif method == 'coherent':
        content = content.replace('#define UNIFORM_GRID 0', '#define UNIFORM_GRID 1')
        content = content.replace('#define COHERENT_GRID 0', '#define COHERENT_GRID 1')
    
    with open(main_cpp_path, 'w') as f:
        f.write(content)
    
    print("Compiling...")
    # Run cmake build
    result = subprocess.run(['cmake', '--build', '../build', '--config', 'Release'],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        return False
    
    return True

def run_benchmark(num_boids, method, visualize=False):
    """Run a single benchmark configuration"""
    print(f"\n{'='*60}")
    print(f"Running: {method.upper()} with N={num_boids}, visualize={visualize}")
    print(f"{'='*60}")
    
    # Compile with settings
    if not compile_with_settings(visualize, method):
        print("Skipping due to compilation error")
        return False
    
    # Check if executable exists
    exe_path = Path(EXECUTABLE)
    if not exe_path.exists():
        print(f"Executable not found: {exe_path}")
        return False
    
    # Run the program
    # Note: You may need to add command-line arguments to your program
    # to set N and frame limit
    try:
        print(f"Executing benchmark (will run for {FRAMES_PER_TEST} frames)...")
        result = subprocess.run([str(exe_path)], 
                              timeout=60,  # 60 second timeout
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            print(f"✓ Benchmark completed successfully")
            return True
        else:
            print(f"✗ Benchmark failed with return code {result.returncode}")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Benchmark timed out")
        return False
    except Exception as e:
        print(f"✗ Error running benchmark: {e}")
        return False

def restore_backup():
    """Restore original main.cpp"""
    main_cpp_path = Path('../src/main.cpp')
    backup_path = main_cpp_path.with_suffix('.cpp.backup')
    
    if backup_path.exists():
        with open(backup_path, 'r') as f:
            content = f.read()
        with open(main_cpp_path, 'w') as f:
            f.write(content)
        print("Restored original main.cpp")

def main():
    """Run all benchmarks"""
    print("="*60)
    print("CUDA Boids Automated Benchmark Suite")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Boid counts: {BOID_COUNTS}")
    print(f"  Methods: {METHODS}")
    print(f"  Frames per test: {FRAMES_PER_TEST}")
    print(f"  Executable: {EXECUTABLE}")
    
    input("\nPress Enter to start benchmarks (or Ctrl+C to cancel)...")
    
    results = []
    
    try:
        for num_boids in BOID_COUNTS:
            for method in METHODS:
                # Run without visualization for accurate performance
                success = run_benchmark(num_boids, method, visualize=False)
                results.append({
                    'num_boids': num_boids,
                    'method': method,
                    'success': success
                })
                
                # Small delay between tests
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nBenchmarks interrupted by user")
    
    finally:
        # Restore original files
        restore_backup()
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['method']:10s} N={result['num_boids']:6d}")
    
    print("\n" + "="*60)
    print("Benchmarks complete!")
    print("Run 'python analyze_performance.py' to analyze results")
    print("="*60)

if __name__ == '__main__':
    main()
