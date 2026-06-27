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
from pathlib import Path

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 11

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
    print("="*70)

    for method in ALL_METHODS:
        print(f"\n  [{METHOD_LABELS[method]}]")
        for vis in [True, False]:
            vis_str = "vis" if vis else "novis"
            ns = sorted(data[method][vis].keys())
            if not ns:
                continue
            print(f"\n    {vis_str}:")
            for n in ns:
                s = data[method][vis][n]
                fps = s.get('mean_fps', 0)
                ms = s.get('mean_total_step_ms', 0)
                print(f"      N={n:7d}  {fps:8.2f} FPS  {ms:8.3f} ms/step")

    plot_experiment_1_fps(data)
    plot_experiment_1_kernel(data)


def plot_experiment_1_fps(data):
    """FPS vs N: 2-panel (with/without vis), all three methods on each panel."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Experiment 1: FPS vs Boid Count (N)', fontsize=14, fontweight='bold')

    for ax, vis, title in [(ax1, True, 'With Visualization'), (ax2, False, 'Without Visualization')]:
        for method in ALL_METHODS:
            ns = sorted(data[method][vis].keys())
            if not ns:
                continue
            fps_vals = [data[method][vis][n].get('mean_fps', 0) for n in ns]
            ax.plot(ns, fps_vals,
                    marker=METHOD_MARKERS[method],
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                    linewidth=2, markersize=8)

        ax.set_xlabel('Number of Boids (N)', fontweight='bold')
        ax.set_ylabel('FPS', fontweight='bold')
        ax.set_title(title, fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.tight_layout()
    out = str(Path(RESULTS_DIR) / 'exp1_fps_vs_n.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {out}")
    plt.close()


def plot_experiment_1_kernel(data):
    """Step time vs N for the no-vis case, all three methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Experiment 1: Step Time vs N (No Visualization)', fontsize=14, fontweight='bold')

    for ax, method in zip(axes, ALL_METHODS):
        ns_novis = sorted(data[method][False].keys())
        if not ns_novis:
            ax.set_visible(False)
            continue
        ms_vals = [data[method][False][n].get('mean_total_step_ms', 0) for n in ns_novis]
        vel_vals = [data[method][False][n].get('mean_kern_update_velocity_ms', 0) for n in ns_novis]

        ax.plot(ns_novis, ms_vals, 'o-', color=METHOD_COLORS[method],
                label='Total step', linewidth=2, markersize=8)
        ax.plot(ns_novis, vel_vals, 's--', color=METHOD_COLORS[method],
                alpha=0.6, label='Velocity kernel', linewidth=1.5, markersize=6)
        ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight='bold')
        ax.set_xlabel('N', fontweight='bold')
        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = str(Path(RESULTS_DIR) / 'exp1_steptime_vs_n.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_experiment_1_comparison(data):
    """Side-by-side FPS at common N values, both vis modes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Experiment 1: Algorithm Comparison at Each N', fontsize=14, fontweight='bold')

    for ax, vis, title in [(axes[0], True, 'With Visualization'), (axes[1], False, 'Without Visualization')]:
        # Collect common N values across all methods
        all_ns = set()
        for m in ALL_METHODS:
            all_ns.update(data[m][vis].keys())
        common_ns = sorted(all_ns)
        if not common_ns:
            continue

        x = np.arange(len(common_ns))
        width = 0.25
        for i, method in enumerate(ALL_METHODS):
            fps_vals = [data[method][vis].get(n, {}).get('mean_fps', 0) for n in common_ns]
            ax.bar(x + i * width, fps_vals, width,
                   label=METHOD_LABELS[method],
                   color=METHOD_COLORS[method], alpha=0.85)

        ax.set_xlabel('Number of Boids (N)', fontweight='bold')
        ax.set_ylabel('FPS', fontweight='bold')
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(n) for n in common_ns], rotation=30, ha='right')
        ax.legend()
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
    fig.suptitle('Experiment 2: Effect of Block Size on Performance', fontsize=14, fontweight='bold')

    ax_time, ax_fps, ax_rel, ax_bar = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    # Panel 1: step time vs block size
    for method in ALL_METHODS:
        bss = sorted(data[method].keys())
        if not bss:
            continue
        ms_vals = [data[method][bs].get('mean_total_step_ms', 0) for bs in bss]
        ax_time.plot(bss, ms_vals,
                     marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                     label=METHOD_LABELS[method], linewidth=2, markersize=8)

    ax_time.set_xlabel('Block Size', fontweight='bold')
    ax_time.set_ylabel('Step Time (ms)', fontweight='bold')
    ax_time.set_title('Step Time vs Block Size', fontsize=12)
    ax_time.set_xscale('log', base=2)
    ax_time.legend()
    ax_time.grid(alpha=0.3)

    # Panel 2: FPS vs block size
    for method in ALL_METHODS:
        bss = sorted(data[method].keys())
        if not bss:
            continue
        fps_vals = [data[method][bs].get('mean_fps', 0) for bs in bss]
        ax_fps.plot(bss, fps_vals,
                    marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method], linewidth=2, markersize=8)

    ax_fps.set_xlabel('Block Size', fontweight='bold')
    ax_fps.set_ylabel('FPS', fontweight='bold')
    ax_fps.set_title('FPS vs Block Size', fontsize=12)
    ax_fps.set_xscale('log', base=2)
    ax_fps.legend()
    ax_fps.grid(alpha=0.3)

    # Panel 3: relative performance (normalized per method)
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
    ax_rel.set_ylabel('Relative Performance (% of best)', fontweight='bold')
    ax_rel.set_title('Relative Performance vs Block Size', fontsize=12)
    ax_rel.set_xscale('log', base=2)
    ax_rel.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_rel.legend()
    ax_rel.grid(alpha=0.3)

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
        width = 0.8 / len(kernel_names)
        bar_colors = plt.cm.tab10(np.linspace(0, 0.9, len(kernel_names)))

        bottoms = np.zeros(len(methods_with_data))
        for ki, (kname, col) in enumerate(col_map.items()):
            vals = [data[m][ref_bs].get(col, 0) for m in methods_with_data]
            ax_bar.bar(x, vals, width=0.6, bottom=bottoms,
                       label=kname, color=bar_colors[ki], alpha=0.85)
            bottoms += np.array(vals)

        ax_bar.set_xlabel('Method', fontweight='bold')
        ax_bar.set_ylabel('Time (ms)', fontweight='bold')
        ax_bar.set_title(f'Kernel Breakdown at BlockSize={ref_bs}', fontsize=12)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([METHOD_LABELS[m] for m in methods_with_data], rotation=10)
        ax_bar.legend(fontsize=8, loc='upper right')
        ax_bar.grid(axis='y', alpha=0.3)

    plt.tight_layout()
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
