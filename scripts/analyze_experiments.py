#!/usr/bin/env python3
"""
Analyze experiment results and generate comprehensive plots for all three methods:
  - Naive brute-force
  - Scattered uniform grid
  - Coherent uniform grid
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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
METHOD_LABELS = {"naive": "Naive (O(N^2))", "scattered": "Scattered Grid", "coherent": "Coherent Grid"}
METHOD_MARKERS = {"naive": "o", "scattered": "s", "coherent": "^"}


# =====================================
# Data Loading
# =====================================

def load_csv(filepath):
    """Load a single CSV, returning a DataFrame or None."""
    try:
        df = pd.read_csv(filepath)
        # Drop first 30 frames as warmup
        return df.iloc[30:].reset_index(drop=True) if len(df) > 30 else df
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def aggregate_stats(df):
    """Return mean stats for a DataFrame."""
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
    """Load all exp1_* CSV files, return nested dict: [method][vis][N] -> stats"""
    pattern = str(Path(RESULTS_DIR) / "exp1_*.csv")
    files = glob.glob(pattern)
    print(f"\nFound {len(files)} Experiment-1 CSV files")

    data = {m: {True: {}, False: {}} for m in ALL_METHODS}

    for fp in sorted(files):
        stem = Path(fp).stem  # exp1_naive_N5000_B128_novis
        parts = stem.split('_')
        try:
            method = parts[1]
            n = int(parts[2][1:])      # N5000 -> 5000
            vis = parts[-1] == 'vis'   # vis / novis
            df = load_csv(fp)
            stats = aggregate_stats(df)
            if stats and method in data:
                data[method][vis][n] = stats
                print(f"  Loaded: {stem}  (frames={len(df) if df is not None else 0})")
        except Exception as e:
            print(f"  Skip {stem}: {e}")

    return data


def load_experiment_2():
    """Load all exp2_* CSV files, return nested dict: [method][block_size] -> stats"""
    pattern = str(Path(RESULTS_DIR) / "exp2_*.csv")
    files = glob.glob(pattern)
    print(f"Found {len(files)} Experiment-2 CSV files")

    data = {m: {} for m in ALL_METHODS}

    for fp in sorted(files):
        stem = Path(fp).stem  # exp2_coherent_N10000_B128_novis
        parts = stem.split('_')
        try:
            method = parts[1]
            block_size = int(parts[3][1:])  # B128 -> 128
            df = load_csv(fp)
            stats = aggregate_stats(df)
            if stats and method in data:
                data[method][block_size] = stats
                print(f"  Loaded: {stem}  (frames={len(df) if df is not None else 0})")
        except Exception as e:
            print(f"  Skip {stem}: {e}")

    return data


# =====================================
# Experiment 1 Analysis
# =====================================

def analyze_experiment_1(data):
    print("\n" + "="*70)
    print("EXPERIMENT 1: Effect of Boid Count (N)")
    print("  Controlled Variable: Block Size = 128, Mode = Headless")
    print("="*70)

    for method in ALL_METHODS:
        print(f"\n  [{METHOD_LABELS[method]}]")
        ns = sorted(data[method][False].keys())  # no-vis only
        if not ns:
            continue
        for n in ns:
            s = data[method][False][n]
            fps = s.get('mean_fps', 0)
            ms = s.get('mean_total_step_ms', 0)
            print(f"    N={n:7d}  {fps:8.2f} FPS  {ms:8.3f} ms/step")

    plot_experiment_1_fps(data)
    plot_experiment_1_kernel(data)


def plot_experiment_1_fps(data):
    """FPS vs N for all three methods (no visualization)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle('Experiment 1: Frame Rate (FPS) vs Boid Count (N)', fontsize=14, fontweight='bold')
    ax.set_title('Controlled Variable: Block Size = 128, Mode = Headless (No Visualization)', fontsize=11, style='italic', pad=10)

    for method in ALL_METHODS:
        ns = sorted(data[method][False].keys())
        if not ns:
            continue
        fps_vals = [data[method][False][n].get('mean_fps', 0) for n in ns]
        ax.plot(ns, fps_vals,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=2, markersize=8)

    ax.set_xlabel('Number of Boids (N)', fontweight='bold')
    ax.set_ylabel('FPS (Frames Per Second)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    out = str(Path(RESULTS_DIR) / 'exp1_fps_vs_n.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {out}")
    plt.close()


def plot_experiment_1_kernel(data):
    """Step time vs N for the no-vis case, with aligned axes and controlled variables."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Experiment 1: Simulation Step Execution Time vs Boid Count (N)', fontsize=15, fontweight='bold')
    plt.figtext(0.5, 0.94, 'Controlled Variable: Block Size = 128, Mode = Headless (No Visualization)', 
                ha='center', fontsize=11, style='italic')

    # Top plot: Direct overlay of all 3 methods on identical axes
    ax_top = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    for method in ALL_METHODS:
        ns = sorted(data[method][False].keys())
        if not ns:
            continue
        ms_vals = [data[method][False][n].get('mean_total_step_ms', 0) for n in ns]
        ax_top.plot(ns, ms_vals, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method], linewidth=2.5, markersize=8)

    ax_top.set_title('Direct Method Comparison: Total Step Time vs N', fontsize=12, fontweight='bold')
    ax_top.set_xlabel('Number of Boids (N)', fontweight='bold')
    ax_top.set_ylabel('Total Step Time (ms)', fontweight='bold')
    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.legend(loc='upper left')
    ax_top.grid(True, which="both", ls="--", alpha=0.3)

    # Bottom 3 plots: Breakdown for each method sharing identical X and Y axes
    first_ax = None
    for i, method in enumerate(ALL_METHODS):
        if first_ax is None:
            ax_b = plt.subplot2grid((2, 3), (1, i))
            first_ax = ax_b
        else:
            ax_b = plt.subplot2grid((2, 3), (1, i), sharex=first_ax, sharey=first_ax)

        ns_novis = sorted(data[method][False].keys())
        if not ns_novis:
            ax_b.set_visible(False)
            continue
        ms_vals = [data[method][False][n].get('mean_total_step_ms', 0) for n in ns_novis]
        vel_vals = [data[method][False][n].get('mean_kern_update_velocity_ms', 0) for n in ns_novis]

        ax_b.plot(ns_novis, ms_vals, 'o-', color=METHOD_COLORS[method],
                label='Total step', linewidth=2, markersize=7)
        ax_b.plot(ns_novis, vel_vals, 's--', color=METHOD_COLORS[method],
                alpha=0.6, label='Velocity kernel', linewidth=1.5, markersize=5)
        ax_b.set_title(f'Breakdown: {METHOD_LABELS[method]}', fontsize=11, fontweight='bold')
        ax_b.set_xlabel('Number of Boids (N)', fontweight='bold')
        ax_b.set_ylabel('Time (ms)', fontweight='bold')
        ax_b.set_xscale('log')
        ax_b.set_yscale('log')
        ax_b.legend(fontsize=9, loc='upper left')
        ax_b.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = str(Path(RESULTS_DIR) / 'exp1_steptime_vs_n.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_experiment_1_comparison(data):
    """Side-by-side FPS comparison at common N values (no visualization)."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle('Experiment 1: Algorithm FPS Comparison at Each N', fontsize=14, fontweight='bold')
    ax.set_title('Controlled Variable: Block Size = 128, Mode = Headless (No Visualization)', fontsize=11, style='italic', pad=10)

    # Collect common N values across all methods
    all_ns = set()
    for m in ALL_METHODS:
        all_ns.update(data[m][False].keys())
    common_ns = sorted(all_ns)
    if not common_ns:
        plt.close()
        return

    x = np.arange(len(common_ns))
    width = 0.25
    for i, method in enumerate(ALL_METHODS):
        fps_vals = [data[method][False].get(n, {}).get('mean_fps', 0) for n in common_ns]
        ax.bar(x + i * width, fps_vals, width,
               label=METHOD_LABELS[method],
               color=METHOD_COLORS[method], alpha=0.85)

    ax.set_xlabel('Number of Boids (N)', fontweight='bold')
    ax.set_ylabel('FPS (Frames Per Second)', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(n) for n in common_ns], rotation=30, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    out = str(Path(RESULTS_DIR) / 'exp1_comparison_bar.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


# =====================================
# Experiment 2 Analysis
# =====================================

def analyze_experiment_2(data):
    print("\n" + "="*70)
    print("EXPERIMENT 2: Effect of Block Size")
    print("  Controlled Variable: Boid Count N = 10,000, Mode = Headless")
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
            print(f"    -> Best block size: {best_bs}")

    plot_experiment_2(data)


def plot_experiment_2(data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment 2: Effect of Block Size on Performance', fontsize=15, fontweight='bold')
    plt.figtext(0.5, 0.95, 'Controlled Variable: Boid Count N = 10,000, Mode = Headless (No Visualization)', 
                ha='center', fontsize=11, style='italic')

    ax_time, ax_speedup, ax_rel, ax_bar = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    # Panel 1: Step Time (ms) vs Block Size
    for method in ALL_METHODS:
        bss = sorted(data[method].keys())
        if not bss:
            continue
        ms_vals = [data[method][bs].get('mean_total_step_ms', 0) for bs in bss]
        ax_time.plot(bss, ms_vals,
                     marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                     label=METHOD_LABELS[method], linewidth=2, markersize=8)

    ax_time.set_xlabel('Block Size', fontweight='bold')
    ax_time.set_ylabel('Total Step Time (ms)', fontweight='bold')
    ax_time.set_title('Absolute Step Time vs Block Size', fontsize=12, fontweight='bold')
    ax_time.set_xscale('log', base=2)
    ax_time.set_xticks([32, 64, 128, 256, 512, 1024])
    ax_time.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_time.legend(loc='upper left')
    ax_time.grid(True, which="both", ls="--", alpha=0.3)

    # Panel 2: Speedup relative to Naive at each Block Size
    naive_bss = data.get("naive", {})
    for method in ALL_METHODS:
        if method == "naive":
            continue
        bss = sorted(data[method].keys())
        if not bss:
            continue
        speedups = []
        valid_bss = []
        for bs in bss:
            n_time = naive_bss.get(bs, {}).get('mean_total_step_ms', 0)
            m_time = data[method][bs].get('mean_total_step_ms', 0)
            if n_time > 0 and m_time > 0:
                speedups.append(n_time / m_time)
                valid_bss.append(bs)
        ax_speedup.plot(valid_bss, speedups,
                        marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                        label=METHOD_LABELS[method], linewidth=2, markersize=8)

    ax_speedup.set_xlabel('Block Size', fontweight='bold')
    ax_speedup.set_ylabel('Speedup (vs Naive at same BS)', fontweight='bold')
    ax_speedup.set_title('Speedup over Naive vs Block Size', fontsize=12, fontweight='bold')
    ax_speedup.set_xscale('log', base=2)
    ax_speedup.set_xticks([32, 64, 128, 256, 512, 1024])
    ax_speedup.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_speedup.legend(loc='upper right')
    ax_speedup.grid(True, which="both", ls="--", alpha=0.3)

    # Panel 3: Internal Sensitivity (% of method's own best performance)
    for method in ALL_METHODS:
        bss = sorted(data[method].keys())
        if not bss:
            continue
        ms_vals = [data[method][bs].get('mean_total_step_ms', float('inf')) for bs in bss]
        best = min(v for v in ms_vals if v > 0)
        rel = [best / v * 100 for v in ms_vals]
        ax_rel.plot(bss, rel,
                    marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method], linewidth=2, markersize=8)

    ax_rel.set_xlabel('Block Size', fontweight='bold')
    ax_rel.set_ylabel('Sensitivity (% of Method\'s Best)', fontweight='bold')
    ax_rel.set_title('Internal Block Size Sensitivity (Self-Normalized)', fontsize=12, fontweight='bold')
    ax_rel.set_xscale('log', base=2)
    ax_rel.set_xticks([32, 64, 128, 256, 512, 1024])
    ax_rel.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_rel.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_rel.legend(loc='lower left')
    ax_rel.grid(True, which="both", ls="--", alpha=0.3)

    # Panel 4: kernel breakdown at block_size=128 for all methods
    ref_bs = 128
    methods_with_data = [m for m in ALL_METHODS if ref_bs in data[m]]
    if methods_with_data:
        kernel_names = ['vel_update', 'pos_update', 'compute_idx', 'reset_buf',
                        'identify_cell', 'thrust_sort', 'reshuffle']
        col_map = {
            'vel_update':    'mean_kern_update_velocity_ms',
            'pos_update':    'mean_kern_update_pos_ms',
            'compute_idx':   'mean_kern_compute_indices_ms',
            'reset_buf':     'mean_kern_reset_buffer_ms',
            'identify_cell': 'mean_kern_identify_cell_ms',
            'thrust_sort':   'mean_thrust_sort_ms',
            'reshuffle':     'mean_kern_reshuffle_ms',
        }
        x = np.arange(len(methods_with_data))
        bar_colors = plt.cm.tab10(np.linspace(0, 0.9, len(kernel_names)))

        bottoms = np.zeros(len(methods_with_data))
        for ki, (kname, col) in enumerate(col_map.items()):
            vals = [data[m][ref_bs].get(col, 0) for m in methods_with_data]
            ax_bar.bar(x, vals, width=0.5, bottom=bottoms,
                       label=kname, color=bar_colors[ki], alpha=0.85)
            bottoms += np.array(vals)

        ax_bar.set_xlabel('Method', fontweight='bold')
        ax_bar.set_ylabel('Time (ms)', fontweight='bold')
        ax_bar.set_title(f'Kernel Phase Breakdown at BlockSize={ref_bs}', fontsize=12, fontweight='bold')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([METHOD_LABELS[m] for m in methods_with_data], rotation=10)
        ax_bar.legend(fontsize=8, loc='upper right')
        ax_bar.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = str(Path(RESULTS_DIR) / 'exp2_blocksize_analysis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {out}")
    plt.close()


# =====================================
# Summary Report
# =====================================

def generate_report(exp1_data, exp2_data):
    report_file = str(Path(RESULTS_DIR) / 'analysis_report.txt')
    lines = []
    lines.append("="*70)
    lines.append("CUDA BOIDS PERFORMANCE ANALYSIS REPORT")
    lines.append("="*70)

    lines.append("\nEXPERIMENT 1: Effect of Boid Count (N)")
    lines.append("  Controlled Variable: Block Size = 128, Mode = Headless")
    lines.append("-"*70)
    for method in ALL_METHODS:
        lines.append(f"\n  {METHOD_LABELS[method]}:")
        ns = sorted(exp1_data[method][False].keys())  # no-vis
        for n in ns:
            s = exp1_data[method][False][n]
            fps = s.get('mean_fps', 0)
            ms = s.get('mean_total_step_ms', 0)
            lines.append(f"    N={n:7d}  {fps:8.2f} FPS  {ms:8.3f} ms/step")

    lines.append("\n\nEXPERIMENT 2: Effect of Block Size")
    lines.append("  Controlled Variable: Boid Count N = 10,000, Mode = Headless")
    lines.append("-"*70)
    for method in ALL_METHODS:
        lines.append(f"\n  {METHOD_LABELS[method]}:")
        bss = sorted(exp2_data[method].keys())
        for bs in bss:
            s = exp2_data[method][bs]
            fps = s.get('mean_fps', 0)
            ms = s.get('mean_total_step_ms', 0)
            lines.append(f"    B={bs:4d}  {fps:8.2f} FPS  {ms:8.3f} ms/step")
        if bss:
            best = min(bss, key=lambda b: exp2_data[method][b].get('mean_total_step_ms', float('inf')))
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
    exp2_data = load_experiment_2()

    has_exp1 = any(exp1_data[m][v] for m in ALL_METHODS for v in [True, False])
    has_exp2 = any(exp2_data[m] for m in ALL_METHODS)

    if has_exp1:
        analyze_experiment_1(exp1_data)
        plot_experiment_1_comparison(exp1_data)
    else:
        print("\nNo Experiment 1 data found.")

    if has_exp2:
        analyze_experiment_2(exp2_data)
    else:
        print("\nNo Experiment 2 data found.")

    if has_exp1 or has_exp2:
        generate_report(exp1_data, exp2_data)

    print("\n" + "="*70)
    print(f"Analysis complete! Results in: {RESULTS_DIR}/")
    print("="*70)


if __name__ == '__main__':
    main()
