#!/usr/bin/env python3
"""
CUDA Boids Performance Analysis — 3 Focused Experiments

  Experiment 1: Algorithm Complexity & Scalability (Step Time vs N, Log-Log)
  Experiment 2: Memory Coalescing Proof (Kernel Phase Breakdown, Stacked Bar)
  Experiment 3: Hardware Execution Configuration (Step Time vs Block Size, Linear)
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import glob
import json
import sys
from pathlib import Path

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 11

# Accept optional directory argument; fall back to root results directory
if len(sys.argv) > 1:
    RESULTS_DIR = sys.argv[1]
else:
    RESULTS_DIR = "../results"

ALL_METHODS = ["naive", "scattered", "coherent"]
METHOD_COLORS = {"naive": "#e74c3c", "scattered": "#3498db", "coherent": "#2ecc71"}
METHOD_LABELS = {"naive": "Naive O(N^2)", "scattered": "Scattered Grid", "coherent": "Coherent Grid"}
METHOD_MARKERS = {"naive": "o", "scattered": "s", "coherent": "^"}


# =====================================
# Data Loading
# =====================================

def load_csv(filepath):
    """Load a single CSV, dropping the first 30 frames as warmup."""
    try:
        df = pd.read_csv(filepath)
        return df.iloc[30:].reset_index(drop=True) if len(df) > 30 else df
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def aggregate_stats(df):
    """Return mean/std stats for a DataFrame."""
    if df is None or len(df) == 0:
        return None
    cols = ['fps', 'frame_ms', 'total_step_ms',
            'kern_update_velocity_ms', 'kern_update_pos_ms',
            'kern_compute_indices_ms', 'kern_reset_buffer_ms',
            'kern_identify_cell_ms', 'thrust_sort_ms', 'kern_reshuffle_ms']
    stats = {}
    for col in cols:
        if col in df.columns:
            stats[f'mean_{col}'] = df[col].mean()
            stats[f'std_{col}'] = df[col].std()
    return stats


def load_experiment_1():
    """Load exp1_* CSVs -> nested dict: [method][N] -> stats"""
    pattern = str(Path(RESULTS_DIR) / "exp1_*.csv")
    files = glob.glob(pattern)
    print(f"\nFound {len(files)} Experiment-1 CSV files")
    data = {m: {} for m in ALL_METHODS}
    for fp in sorted(files):
        stem = Path(fp).stem
        parts = stem.split('_')
        try:
            method = parts[1]
            n = int(parts[2][1:])
            df = load_csv(fp)
            stats = aggregate_stats(df)
            if stats and method in data:
                data[method][n] = stats
                print(f"  Loaded: {stem}  (frames={len(df) if df is not None else 0})")
        except Exception as e:
            print(f"  Skip {stem}: {e}")
    return data


def load_experiment_3():
    """Load exp3_* (or legacy exp2_*) CSVs -> nested dict: [method][block_size] -> stats"""
    # Try exp3_ first, fall back to exp2_ for backwards compat
    pattern = str(Path(RESULTS_DIR) / "exp3_*.csv")
    files = glob.glob(pattern)
    if not files:
        pattern = str(Path(RESULTS_DIR) / "exp2_*.csv")
        files = glob.glob(pattern)
    print(f"Found {len(files)} Experiment-3 CSV files")
    data = {m: {} for m in ALL_METHODS}
    for fp in sorted(files):
        stem = Path(fp).stem
        parts = stem.split('_')
        try:
            method = parts[1]
            block_size = int(parts[3][1:])
            df = load_csv(fp)
            stats = aggregate_stats(df)
            if stats and method in data:
                data[method][block_size] = stats
                print(f"  Loaded: {stem}  (frames={len(df) if df is not None else 0})")
        except Exception as e:
            print(f"  Skip {stem}: {e}")
    return data


# =====================================================================
# Experiment 1: Algorithm Complexity & Scalability
#   X = N (log), Y = Total Step Time ms (log), one line per algorithm
# =====================================================================

def analyze_experiment_1(data):
    print("\n" + "="*70)
    print("EXPERIMENT 1: Algorithm Complexity & Scalability")
    print("  Control: Block Size = 128, Headless Mode")
    print("="*70)
    for method in ALL_METHODS:
        ns = sorted(data[method].keys())
        if not ns:
            continue
        print(f"\n  [{METHOD_LABELS[method]}]")
        for n in ns:
            s = data[method][n]
            fps = s.get('mean_fps', 0)
            ms = s.get('mean_total_step_ms', 0)
            print(f"    N={n:7d}  {fps:8.2f} FPS  {ms:10.3f} ms/step")
    plot_experiment_1(data)


def plot_experiment_1(data):
    """Single clean Log-Log chart: Total Step Time vs N."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle('Experiment 1: Algorithm Complexity & Scalability',
                 fontsize=15, fontweight='bold')
    ax.set_title('Control: Block Size = 128, Headless Mode',
                 fontsize=11, style='italic', pad=8)

    for method in ALL_METHODS:
        ns = sorted(data[method].keys())
        if not ns:
            continue
        ms_vals = [data[method][n].get('mean_total_step_ms', 0) for n in ns]
        ax.plot(ns, ms_vals,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=2.5, markersize=8)

    ax.set_xlabel('Number of Boids (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Step Time (ms)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    out = str(Path(RESULTS_DIR) / 'fig1_steptime_vs_n.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {out}")
    plt.close()


# =====================================================================
# Experiment 2: Memory Coalescing Proof (Kernel Phase Breakdown)
#   Stacked bar chart at a representative large N, Block Size = 128
# =====================================================================

def analyze_experiment_2(exp1_data):
    """
    Extract kernel breakdown from Experiment 1 data at the largest common N
    where all three methods have data, to prove memory coalescing benefit.
    """
    # Find the largest N where all 3 methods have data
    common_ns = set(exp1_data["naive"].keys()) & set(exp1_data["scattered"].keys()) & set(exp1_data["coherent"].keys())
    if not common_ns:
        # Fall back: use largest N where at least scattered + coherent exist
        common_ns = set(exp1_data["scattered"].keys()) & set(exp1_data["coherent"].keys())
    if not common_ns:
        print("\n  No common N values found for kernel breakdown. Skipping Experiment 2.")
        return

    target_n = max(common_ns)

    # Also gather a secondary N for multi-scale comparison
    large_ns = sorted([n for n in (set(exp1_data["scattered"].keys()) & set(exp1_data["coherent"].keys())) if n > target_n], reverse=False)

    print("\n" + "="*70)
    print(f"EXPERIMENT 2: Kernel Phase Breakdown (Memory Coalescing Proof)")
    print(f"  Control: Block Size = 128, Headless Mode")
    print(f"  Representative N = {target_n:,}")
    print("="*70)

    # Print detailed breakdown
    kernel_cols = {
        'Velocity Update': 'mean_kern_update_velocity_ms',
        'Position Update': 'mean_kern_update_pos_ms',
        'Compute Indices': 'mean_kern_compute_indices_ms',
        'Reset Buffers':   'mean_kern_reset_buffer_ms',
        'Identify Cells':  'mean_kern_identify_cell_ms',
        'Thrust Sort':     'mean_thrust_sort_ms',
        'Reshuffle Data':  'mean_kern_reshuffle_ms',
    }

    methods_at_n = [m for m in ALL_METHODS if target_n in exp1_data[m]]
    for method in methods_at_n:
        s = exp1_data[method][target_n]
        total = s.get('mean_total_step_ms', 0)
        print(f"\n  [{METHOD_LABELS[method]}]  Total = {total:.3f} ms")
        for kname, col in kernel_cols.items():
            val = s.get(col, 0)
            pct = (val / total * 100) if total > 0 else 0
            print(f"    {kname:20s}  {val:8.4f} ms  ({pct:5.1f}%)")

    # Also print for larger Ns (scattered vs coherent only) if available
    if large_ns:
        extra_n = large_ns[-1]  # largest available
        print(f"\n  --- Additional scale: N = {extra_n:,} (Grid methods only) ---")
        for method in ["scattered", "coherent"]:
            if extra_n in exp1_data[method]:
                s = exp1_data[method][extra_n]
                total = s.get('mean_total_step_ms', 0)
                print(f"\n  [{METHOD_LABELS[method]}]  Total = {total:.3f} ms")
                for kname, col in kernel_cols.items():
                    val = s.get(col, 0)
                    pct = (val / total * 100) if total > 0 else 0
                    print(f"    {kname:20s}  {val:8.4f} ms  ({pct:5.1f}%)")

    plot_experiment_2(exp1_data, target_n, large_ns)


def plot_experiment_2(exp1_data, target_n, extra_ns):
    """Stacked bar chart showing kernel phase breakdown."""
    kernel_phases = [
        ('Velocity Update',  'mean_kern_update_velocity_ms',  '#e74c3c'),
        ('Thrust Sort',      'mean_thrust_sort_ms',           '#f39c12'),
        ('Reshuffle Data',   'mean_kern_reshuffle_ms',        '#9b59b6'),
        ('Compute Indices',  'mean_kern_compute_indices_ms',  '#3498db'),
        ('Identify Cells',   'mean_kern_identify_cell_ms',    '#1abc9c'),
        ('Reset Buffers',    'mean_kern_reset_buffer_ms',     '#95a5a6'),
        ('Position Update',  'mean_kern_update_pos_ms',       '#2ecc71'),
    ]

    # Determine which N values to show
    # Always show the largest common N (all 3 methods)
    # Optionally show 1-2 larger Ns for grid-only comparison
    chart_groups = []  # list of (label, [methods_with_data])

    methods_at_target = [m for m in ALL_METHODS if target_n in exp1_data[m]]
    chart_groups.append((f"N={target_n:,}", target_n, methods_at_target))

    # Add larger scales where scattered+coherent both exist
    grid_extra = [n for n in extra_ns if n in exp1_data["scattered"] and n in exp1_data["coherent"]]
    # Pick up to 2 representative larger Ns
    if grid_extra:
        picks = []
        if len(grid_extra) >= 2:
            picks = [grid_extra[len(grid_extra)//2], grid_extra[-1]]
        else:
            picks = grid_extra[:1]
        for n in picks:
            methods_here = [m for m in ALL_METHODS if n in exp1_data[m]]
            chart_groups.append((f"N={n:,}", n, methods_here))

    # Build bar labels and data
    bar_labels = []
    bar_data = []   # each entry: dict of kernel_col -> value
    for group_label, n_val, methods in chart_groups:
        for method in methods:
            bar_labels.append(f"{METHOD_LABELS[method]}\n({group_label})")
            bar_data.append(exp1_data[method][n_val])

    n_bars = len(bar_labels)
    if n_bars == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, n_bars * 1.8), 7))
    fig.suptitle('Experiment 2: Kernel Phase Breakdown — Memory Coalescing Proof',
                 fontsize=14, fontweight='bold')
    ax.set_title('Control: Block Size = 128, Headless Mode',
                 fontsize=11, style='italic', pad=8)

    x = np.arange(n_bars)
    bottoms = np.zeros(n_bars)

    for kname, col, color in kernel_phases:
        vals = np.array([d.get(col, 0) for d in bar_data])
        bars = ax.bar(x, vals, width=0.55, bottom=bottoms,
                      label=kname, color=color, alpha=0.88, edgecolor='white', linewidth=0.5)
        # Annotate dominant phases (> 15% of total)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            total = bar_data[i].get('mean_total_step_ms', 1)
            pct = v / total * 100 if total > 0 else 0
            if pct > 15 and v > 0.01:
                ax.text(x[i], b + v / 2, f'{v:.2f}ms\n({pct:.0f}%)',
                        ha='center', va='center', fontsize=7.5, fontweight='bold', color='white')
        bottoms += vals

    # Add total labels on top
    for i, total_h in enumerate(bottoms):
        ax.text(x[i], total_h + total_h * 0.02, f'{total_h:.2f} ms',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Algorithm & Scale', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9, ha='center')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, ls='--')

    plt.tight_layout()
    out = str(Path(RESULTS_DIR) / 'fig2_kernel_breakdown.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {out}")
    plt.close()


# =====================================================================
# Experiment 3: Hardware Execution Configuration (Step Time vs Block Size)
#   X = Block Size (linear), Y = Total Step Time ms, one line per algo
# =====================================================================

def analyze_experiment_3(data):
    print("\n" + "="*70)
    print("EXPERIMENT 3: Hardware Execution Configuration (Block Size)")
    print("  Control: N = 10,000, Headless Mode")
    print("="*70)
    for method in ALL_METHODS:
        bss = sorted(data[method].keys())
        if not bss:
            continue
        print(f"\n  [{METHOD_LABELS[method]}]")
        for bs in bss:
            s = data[method][bs]
            fps = s.get('mean_fps', 0)
            ms = s.get('mean_total_step_ms', 0)
            std = s.get('std_total_step_ms', 0)
            print(f"    B={bs:4d}  {fps:8.2f} FPS  {ms:8.3f} +/- {std:6.3f} ms")
        if bss:
            best_bs = min(bss, key=lambda b: data[method][b].get('mean_total_step_ms', float('inf')))
            print(f"    -> Optimal block size: {best_bs}")
    plot_experiment_3(data)


def plot_experiment_3(data):
    """Single clean linear-scale chart: Total Step Time vs Block Size."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle('Experiment 3: Performance vs Block Size',
                 fontsize=15, fontweight='bold')
    ax.set_title('Control: N = 10,000, Headless Mode',
                 fontsize=11, style='italic', pad=8)

    for method in ALL_METHODS:
        bss = sorted(data[method].keys())
        if not bss:
            continue
        ms_vals = [data[method][bs].get('mean_total_step_ms', 0) for bs in bss]
        std_vals = [data[method][bs].get('std_total_step_ms', 0) for bs in bss]
        ax.errorbar(bss, ms_vals, yerr=std_vals,
                    marker=METHOD_MARKERS[method],
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                    linewidth=2.5, markersize=8, capsize=4, capthick=1.5)

    ax.set_xlabel('Block Size (threads per block)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Step Time (ms)', fontsize=12, fontweight='bold')
    # Linear X axis with explicit tick labels
    ax.set_xticks([32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    out = str(Path(RESULTS_DIR) / 'fig3_steptime_vs_blocksize.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {out}")
    plt.close()


# =====================================================================
# Summary Report
# =====================================================================

def generate_report(exp1_data, exp3_data):
    report_file = str(Path(RESULTS_DIR) / 'analysis_report.txt')
    lines = []
    lines.append("="*70)
    lines.append("CUDA BOIDS PERFORMANCE ANALYSIS REPORT")
    lines.append("="*70)

    # Experiment 1
    lines.append("\nEXPERIMENT 1: Algorithm Complexity & Scalability")
    lines.append("  Control: Block Size = 128, Headless Mode")
    lines.append("-"*70)
    for method in ALL_METHODS:
        lines.append(f"\n  {METHOD_LABELS[method]}:")
        ns = sorted(exp1_data[method].keys())
        for n in ns:
            s = exp1_data[method][n]
            fps = s.get('mean_fps', 0)
            ms = s.get('mean_total_step_ms', 0)
            lines.append(f"    N={n:7d}  {fps:8.2f} FPS  {ms:10.3f} ms/step")

    # Experiment 2 (kernel breakdown from exp1 data)
    common_ns = set(exp1_data["naive"].keys()) & set(exp1_data["scattered"].keys()) & set(exp1_data["coherent"].keys())
    if common_ns:
        target_n = max(common_ns)
        kernel_cols = {
            'Velocity Update': 'mean_kern_update_velocity_ms',
            'Position Update': 'mean_kern_update_pos_ms',
            'Compute Indices': 'mean_kern_compute_indices_ms',
            'Reset Buffers':   'mean_kern_reset_buffer_ms',
            'Identify Cells':  'mean_kern_identify_cell_ms',
            'Thrust Sort':     'mean_thrust_sort_ms',
            'Reshuffle Data':  'mean_kern_reshuffle_ms',
        }
        lines.append(f"\n\nEXPERIMENT 2: Kernel Phase Breakdown (Memory Coalescing Proof)")
        lines.append(f"  Control: Block Size = 128, N = {target_n:,}, Headless Mode")
        lines.append("-"*70)
        methods_at_n = [m for m in ALL_METHODS if target_n in exp1_data[m]]
        for method in methods_at_n:
            s = exp1_data[method][target_n]
            total = s.get('mean_total_step_ms', 0)
            lines.append(f"\n  {METHOD_LABELS[method]}:  Total = {total:.3f} ms")
            for kname, col in kernel_cols.items():
                val = s.get(col, 0)
                pct = (val / total * 100) if total > 0 else 0
                lines.append(f"    {kname:20s}  {val:8.4f} ms  ({pct:5.1f}%)")

    # Experiment 3
    lines.append("\n\nEXPERIMENT 3: Performance vs Block Size")
    lines.append("  Control: N = 10,000, Headless Mode")
    lines.append("-"*70)
    for method in ALL_METHODS:
        lines.append(f"\n  {METHOD_LABELS[method]}:")
        bss = sorted(exp3_data[method].keys())
        for bs in bss:
            s = exp3_data[method][bs]
            fps = s.get('mean_fps', 0)
            ms = s.get('mean_total_step_ms', 0)
            lines.append(f"    B={bs:4d}  {fps:8.2f} FPS  {ms:8.3f} ms/step")
        if bss:
            best = min(bss, key=lambda b: exp3_data[method][b].get('mean_total_step_ms', float('inf')))
            lines.append(f"    Optimal block size: {best}")

    with open(report_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\n  Saved: {report_file}")


# =====================================
# Main
# =====================================

def main():
    print("="*70)
    print("CUDA Boids Experiment Analysis")
    print("="*70)

    exp1_data = load_experiment_1()
    exp3_data = load_experiment_3()

    has_exp1 = any(exp1_data[m] for m in ALL_METHODS)
    has_exp3 = any(exp3_data[m] for m in ALL_METHODS)

    if has_exp1:
        analyze_experiment_1(exp1_data)
        analyze_experiment_2(exp1_data)  # Kernel breakdown extracted from exp1 data
    else:
        print("\nNo Experiment 1 data found.")

    if has_exp3:
        analyze_experiment_3(exp3_data)
    else:
        print("\nNo Experiment 3 data found.")

    if has_exp1 or has_exp3:
        generate_report(exp1_data, exp3_data)

    print("\n" + "="*70)
    print(f"Analysis complete! Results in: {RESULTS_DIR}/")
    print("="*70)


if __name__ == '__main__':
    main()
