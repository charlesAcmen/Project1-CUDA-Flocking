#!/usr/bin/env python3
"""
Generate comprehensive performance report with tables and charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
from datetime import datetime

def load_all_data():
    """Load all CSV files and organize by method and N"""
    csv_files = glob.glob('perf_*.csv')
    
    data = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract method and N from filename
            # Expected format: perf_METHOD_NXXXX.csv
            parts = Path(csv_file).stem.split('_')
            if len(parts) >= 2:
                method = parts[1]
                n_value = 5000  # default
                
                for part in parts:
                    if part.startswith('N'):
                        try:
                            n_value = int(part[1:])
                        except:
                            pass
                
                key = (method, n_value)
                data[key] = df
                
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return data

def create_summary_table(data):
    """Create summary statistics table"""
    rows = []
    
    for (method, n), df in sorted(data.items()):
        if len(df) <= 10:
            continue
        
        df_stable = df.iloc[10:]  # Skip warmup
        
        row = {
            'Method': method.capitalize(),
            'N': n,
            'Avg FPS': f"{df_stable['fps'].mean():.2f}",
            'Std FPS': f"{df_stable['fps'].std():.2f}",
            'Avg Frame (ms)': f"{df_stable['frame_ms'].mean():.3f}",
            'Avg Step (ms)': f"{df_stable['total_step_ms'].mean():.3f}",
            'Velocity Update (ms)': f"{df_stable['kern_update_velocity_ms'].mean():.3f}",
            'Position Update (ms)': f"{df_stable['kern_update_pos_ms'].mean():.3f}",
        }
        
        # Add grid-specific columns if they exist
        if 'thrust_sort_ms' in df_stable.columns:
            thrust_val = df_stable['thrust_sort_ms'].mean()
            if thrust_val > 0:
                row['Thrust Sort (ms)'] = f"{thrust_val:.3f}"
        
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    return summary_df

def plot_n_scaling(data, output_file='n_scaling.png'):
    """Plot performance scaling with N"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = {}
    for (method, n), df in data.items():
        if method not in methods:
            methods[method] = {'n': [], 'fps': [], 'time': [], 'vel_time': []}
        
        if len(df) > 10:
            df_stable = df.iloc[10:]
            methods[method]['n'].append(n)
            methods[method]['fps'].append(df_stable['fps'].mean())
            methods[method]['time'].append(df_stable['total_step_ms'].mean())
            methods[method]['vel_time'].append(df_stable['kern_update_velocity_ms'].mean())
    
    colors = {'naive': '#1f77b4', 'scattered': '#ff7f0e', 'coherent': '#2ca02c'}
    markers = {'naive': 'o', 'scattered': 's', 'coherent': '^'}
    
    # FPS vs N
    for method, values in methods.items():
        if values['n']:
            sorted_data = sorted(zip(values['n'], values['fps']))
            n_vals, fps_vals = zip(*sorted_data)
            ax1.plot(n_vals, fps_vals, 
                    marker=markers.get(method, 'o'),
                    linewidth=2, 
                    label=method.capitalize(),
                    color=colors.get(method))
    
    ax1.set_xlabel('Number of Boids (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('FPS', fontsize=12, fontweight='bold')
    ax1.set_title('FPS vs Boid Count', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log')
    
    # Execution time vs N
    for method, values in methods.items():
        if values['n']:
            sorted_data = sorted(zip(values['n'], values['time']))
            n_vals, time_vals = zip(*sorted_data)
            ax2.plot(n_vals, time_vals, 
                    marker=markers.get(method, 'o'),
                    linewidth=2, 
                    label=method.capitalize(),
                    color=colors.get(method))
    
    ax2.set_xlabel('Number of Boids (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Execution Time vs Boid Count', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Speedup vs N (relative to naive)
    naive_data = methods.get('naive', {})
    if naive_data and naive_data['n']:
        naive_dict = dict(zip(naive_data['n'], naive_data['time']))
        
        for method in ['scattered', 'coherent']:
            if method in methods:
                speedups = []
                n_vals = []
                for n, time in zip(methods[method]['n'], methods[method]['time']):
                    if n in naive_dict:
                        speedup = naive_dict[n] / time
                        speedups.append(speedup)
                        n_vals.append(n)
                
                if speedups:
                    sorted_data = sorted(zip(n_vals, speedups))
                    n_vals, speedup_vals = zip(*sorted_data)
                    ax3.plot(n_vals, speedup_vals, 
                            marker=markers.get(method, 'o'),
                            linewidth=2, 
                            label=method.capitalize(),
                            color=colors.get(method))
        
        ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
        ax3.set_xlabel('Number of Boids (N)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
        ax3.set_title('Speedup vs Boid Count (relative to Naive)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.set_xscale('log')
    
    # Velocity update time vs N
    for method, values in methods.items():
        if values['n']:
            sorted_data = sorted(zip(values['n'], values['vel_time']))
            n_vals, vel_vals = zip(*sorted_data)
            ax4.plot(n_vals, vel_vals, 
                    marker=markers.get(method, 'o'),
                    linewidth=2, 
                    label=method.capitalize(),
                    color=colors.get(method))
    
    ax4.set_xlabel('Number of Boids (N)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Velocity Update Time (ms)', fontsize=12, fontweight='bold')
    ax4.set_title('Velocity Update Time vs Boid Count', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved N-scaling analysis to {output_file}")
    plt.close()

def export_to_excel(summary_df, output_file='performance_report.xlsx'):
    """Export summary to Excel (if openpyxl is available)"""
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Summary']
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        
        print(f"Exported summary to {output_file}")
        return True
        
    except ImportError:
        print("openpyxl not available, skipping Excel export")
        print("Install with: pip install openpyxl")
        return False

def main():
    """Generate comprehensive report"""
    print("="*80)
    print("CUDA Boids Performance Report Generator")
    print("="*80)
    
    data = load_all_data()
    
    if not data:
        print("No performance data found!")
        return
    
    print(f"\nLoaded data for {len(data)} configurations:")
    for (method, n) in sorted(data.keys()):
        print(f"  - {method.capitalize()}: N={n}")
    
    # Create summary table
    print("\nGenerating summary table...")
    summary_df = create_summary_table(data)
    
    # Print to console
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Export to CSV
    summary_csv = 'performance_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to {summary_csv}")
    
    # Export to Excel if possible
    export_to_excel(summary_df)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_n_scaling(data)
    
    print("\n" + "="*80)
    print("Report generation complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - performance_summary.csv")
    print("  - performance_report.xlsx (if openpyxl available)")
    print("  - n_scaling.png")

if __name__ == '__main__':
    main()
