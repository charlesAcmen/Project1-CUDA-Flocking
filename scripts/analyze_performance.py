#!/usr/bin/env python3
"""
Performance Analysis Script for CUDA Boids Flocking Simulation
Analyzes CSV performance data and generates comparison charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path

def load_perf_data(csv_file):
    """Load performance data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None

def analyze_single_run(df, method_name):
    """Analyze a single run and print statistics"""
    if df is None or len(df) == 0:
        return None
    
    # Skip first few frames for warmup
    warmup_frames = 10
    df_stable = df.iloc[warmup_frames:]
    
    stats = {
        'method': method_name,
        'avg_fps': df_stable['fps'].mean(),
        'std_fps': df_stable['fps'].std(),
        'avg_frame_ms': df_stable['frame_ms'].mean(),
        'std_frame_ms': df_stable['frame_ms'].std(),
        'avg_total_step_ms': df_stable['total_step_ms'].mean(),
        'avg_velocity_update_ms': df_stable['kern_update_velocity_ms'].mean(),
        'avg_position_update_ms': df_stable['kern_update_pos_ms'].mean(),
    }
    
    # Add grid-specific metrics if they exist
    if 'kern_compute_indices_ms' in df_stable.columns:
        stats['avg_compute_indices_ms'] = df_stable['kern_compute_indices_ms'].mean()
    if 'thrust_sort_ms' in df_stable.columns:
        stats['avg_thrust_sort_ms'] = df_stable['thrust_sort_ms'].mean()
    if 'kern_identify_cell_ms' in df_stable.columns:
        stats['avg_identify_cell_ms'] = df_stable['kern_identify_cell_ms'].mean()
    if 'kern_reshuffle_ms' in df_stable.columns:
        stats['avg_reshuffle_ms'] = df_stable['kern_reshuffle_ms'].mean()
    
    return stats

def print_statistics(stats_list):
    """Print formatted statistics for all methods"""
    print("\n" + "="*80)
    print("PERFORMANCE STATISTICS (excluding first 10 warmup frames)")
    print("="*80)
    
    for stats in stats_list:
        if stats is None:
            continue
        print(f"\n{stats['method'].upper()}:")
        print(f"  Average FPS: {stats['avg_fps']:.2f} ± {stats['std_fps']:.2f}")
        print(f"  Average Frame Time: {stats['avg_frame_ms']:.3f} ± {stats['std_frame_ms']:.3f} ms")
        print(f"  Average Total Step Time: {stats['avg_total_step_ms']:.3f} ms")
        print(f"  Average Velocity Update: {stats['avg_velocity_update_ms']:.3f} ms")
        print(f"  Average Position Update: {stats['avg_position_update_ms']:.3f} ms")
        
        if 'avg_compute_indices_ms' in stats:
            print(f"  Average Compute Indices: {stats['avg_compute_indices_ms']:.3f} ms")
        if 'avg_thrust_sort_ms' in stats:
            print(f"  Average Thrust Sort: {stats['avg_thrust_sort_ms']:.3f} ms")
        if 'avg_identify_cell_ms' in stats:
            print(f"  Average Identify Cell: {stats['avg_identify_cell_ms']:.3f} ms")
        if 'avg_reshuffle_ms' in stats:
            print(f"  Average Reshuffle: {stats['avg_reshuffle_ms']:.3f} ms")

def plot_fps_comparison(data_dict, output_file='fps_comparison.png'):
    """Plot FPS comparison across methods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = []
    avg_fps = []
    std_fps = []
    
    for method, df in data_dict.items():
        if df is not None and len(df) > 10:
            df_stable = df.iloc[10:]
            methods.append(method)
            avg_fps.append(df_stable['fps'].mean())
            std_fps.append(df_stable['fps'].std())
    
    # Bar chart
    x_pos = np.arange(len(methods))
    ax1.bar(x_pos, avg_fps, yerr=std_fps, capsize=5, alpha=0.7, 
            color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('FPS', fontsize=12)
    ax1.set_title('Average FPS Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.grid(axis='y', alpha=0.3)
    
    # Time series
    for method, df in data_dict.items():
        if df is not None and len(df) > 0:
            ax2.plot(df['frame'], df['fps'], label=method, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('FPS', fontsize=12)
    ax2.set_title('FPS Over Time', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved FPS comparison to {output_file}")
    plt.close()

def plot_kernel_breakdown(data_dict, output_file='kernel_breakdown.png'):
    """Plot kernel execution time breakdown"""
    fig, axes = plt.subplots(1, len(data_dict), figsize=(6*len(data_dict), 5))
    
    if len(data_dict) == 1:
        axes = [axes]
    
    for idx, (method, df) in enumerate(data_dict.items()):
        if df is None or len(df) <= 10:
            continue
        
        df_stable = df.iloc[10:]
        
        # Collect kernel times
        kernels = []
        times = []
        
        if 'kern_update_velocity_ms' in df_stable.columns:
            kernels.append('Velocity Update')
            times.append(df_stable['kern_update_velocity_ms'].mean())
        
        if 'kern_update_pos_ms' in df_stable.columns:
            kernels.append('Position Update')
            times.append(df_stable['kern_update_pos_ms'].mean())
        
        if 'kern_compute_indices_ms' in df_stable.columns:
            val = df_stable['kern_compute_indices_ms'].mean()
            if val > 0:
                kernels.append('Compute Indices')
                times.append(val)
        
        if 'thrust_sort_ms' in df_stable.columns:
            val = df_stable['thrust_sort_ms'].mean()
            if val > 0:
                kernels.append('Thrust Sort')
                times.append(val)
        
        if 'kern_identify_cell_ms' in df_stable.columns:
            val = df_stable['kern_identify_cell_ms'].mean()
            if val > 0:
                kernels.append('Identify Cell')
                times.append(val)
        
        if 'kern_reshuffle_ms' in df_stable.columns:
            val = df_stable['kern_reshuffle_ms'].mean()
            if val > 0:
                kernels.append('Reshuffle Data')
                times.append(val)
        
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(kernels)))
        wedges, texts, autotexts = axes[idx].pie(times, labels=kernels, autopct='%1.1f%%',
                                                   colors=colors, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        axes[idx].set_title(f'{method}\n(Total: {sum(times):.2f} ms)', 
                           fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved kernel breakdown to {output_file}")
    plt.close()

def plot_speedup_analysis(stats_list, output_file='speedup_analysis.png'):
    """Plot speedup comparison relative to naive method"""
    if len(stats_list) < 2:
        print("Need at least 2 methods for speedup analysis")
        return
    
    # Find naive baseline
    naive_stats = None
    for stats in stats_list:
        if stats and 'naive' in stats['method'].lower():
            naive_stats = stats
            break
    
    if naive_stats is None:
        print("No naive baseline found for speedup calculation")
        return
    
    baseline_time = naive_stats['avg_total_step_ms']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = []
    speedups = []
    times = []
    
    for stats in stats_list:
        if stats is None:
            continue
        methods.append(stats['method'])
        times.append(stats['avg_total_step_ms'])
        speedups.append(baseline_time / stats['avg_total_step_ms'])
    
    # Speedup chart
    x_pos = np.arange(len(methods))
    colors = ['#1f77b4' if s < 1 else '#2ca02c' for s in speedups]
    bars = ax1.bar(x_pos, speedups, alpha=0.7, color=colors)
    ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Baseline (Naive)')
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Speedup (x)', fontsize=12)
    ax1.set_title('Speedup Relative to Naive Method', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontweight='bold')
    
    # Execution time chart
    ax2.bar(x_pos, times, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('Execution Time (ms)', fontsize=12)
    ax2.set_title('Average Kernel Execution Time', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (pos, time) in enumerate(zip(x_pos, times)):
        ax2.text(pos, time, f'{time:.2f}ms',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved speedup analysis to {output_file}")
    plt.close()

def main():
    """Main analysis function"""
    print("CUDA Boids Performance Analysis")
    print("="*80)
    
    # Find all CSV files in current directory
    csv_files = glob.glob('perf_*.csv')
    
    if not csv_files:
        print("No performance CSV files found!")
        print("Expected files like: perf_naive_N5000.csv, perf_scattered_N5000.csv, etc.")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f}")
    
    # Load data
    data_dict = {}
    stats_list = []
    
    for csv_file in sorted(csv_files):
        # Extract method name from filename
        method_name = Path(csv_file).stem.replace('perf_', '').split('_N')[0]
        
        df = load_perf_data(csv_file)
        if df is not None:
            data_dict[method_name] = df
            stats = analyze_single_run(df, method_name)
            if stats:
                stats_list.append(stats)
    
    if not stats_list:
        print("No valid data loaded!")
        return
    
    # Print statistics
    print_statistics(stats_list)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    plot_fps_comparison(data_dict)
    plot_kernel_breakdown(data_dict)
    plot_speedup_analysis(stats_list)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()
