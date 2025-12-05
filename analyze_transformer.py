#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# file path
BASELINE_CSV = Path("results/transformer/baseline_results3.csv")
PREDICTION_CSV = Path("results/transformer/prediction_results3.csv")
PRE_CSV = Path("results/neural_network/prediction_results3.csv")
OUTPUT_IMG = "analysis_nn3.png"

def analyze():
    if not BASELINE_CSV.exists() or not PREDICTION_CSV.exists() or not PRE_CSV.exists():
        print("Error: One or more CSV files not found.")
        return

    print("Loading data...")
    df_base = pd.read_csv(BASELINE_CSV)
    df_pred = pd.read_csv(PREDICTION_CSV)
    df_pre  = pd.read_csv(PRE_CSV)

    df_base = df_base.rename(columns={"Status": "Status_Base", "Nodes_Explored": "Nodes_Base"})
    df_pred = df_pred.rename(columns={"Status": "Status_Pred", "Nodes_Explored": "Nodes_Pred"})
    df_pre  = df_pre.rename(columns={"Status": "Status_Pre", "Nodes_Explored": "Nodes_Pre"})
    
    merged = pd.merge(df_base, df_pred, on="Problem_ID", how="outer")
    merged = pd.merge(merged, df_pre, on="Problem_ID", how="outer", suffixes=('', '_Pre'))
    
    print(f"Total Problems merged: {len(merged)}")

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # color
    c_base = '#95a5a6'  # Grey
    c_pre  = '#f39c12'  # Orange
    c_curr = '#3498db'  # Blue
    c_unq  = '#e74c3c'  # Red

    # total solved by each model
    ax1 = axs[0, 0]
    cnt_base = (merged['Status_Base'] == 'Solved').sum()
    cnt_pre  = (merged['Status_Pre'] == 'Solved').sum()
    cnt_curr = (merged['Status_Pred'] == 'Solved').sum()
    
    bars1 = ax1.bar(['Baseline', 'Pre-Train', 'Current'], [cnt_base, cnt_pre, cnt_curr], 
                    color=[c_base, c_pre, c_curr])
    ax1.set_title('Total Problems Solved', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.bar_label(bars1, padding=3)

    # unique solved
    ax2 = axs[0, 1]
    
    m_base = merged['Status_Base'] == 'Solved'
    m_curr = merged['Status_Pred'] == 'Solved'
    m_pre  = merged['Status_Pre'] == 'Solved'

    # transformer solved but baseline failed
    rescue_pre  = len(merged[m_pre & (~m_base)])
    rescue_curr = len(merged[m_curr & (~m_base)])
    # baseline solved but current transformer failed
    base_only   = len(merged[m_base & (~m_curr)])

    bars2 = ax2.bar(['Baseline Only', 'Pre Rescue', 'Current Rescue'], 
                    [base_only, rescue_pre, rescue_curr], 
                    color=[c_unq, c_pre, c_curr])
    ax2.set_title('Unique Solves (Rescue vs Loss)', fontweight='bold')
    ax2.bar_label(bars2, padding=3)
    ax2.text(0.5, 0.92, "Rescue: Model Solved, Base Timeout", transform=ax2.transAxes, ha='center', color='gray')

    # graph
    def plot_scatter(ax, col_model_nodes, col_model_status, color, label_name):
        # baseline and transformer both solved
        mask = (merged['Status_Base'] == 'Solved') & (merged[col_model_status] == 'Solved')
        data = merged[mask]
        
        count = len(data)
        if count == 0:
            ax.text(0.5, 0.5, "No overlapping solved problems", ha='center', transform=ax.transAxes)
            ax.set_title(f"{label_name} vs Baseline")
            return

        x = data['Nodes_Base']
        y = data[col_model_nodes]
        
        model_wins = (y < x).sum()
        base_wins = (x < y).sum()
        
        ax.scatter(x, y, alpha=0.6, c=color, edgecolors='none', s=30)
        
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        limit = [min_val * 0.8, max_val * 1.5]
        
        ax.plot(limit, limit, 'k--', alpha=0.5, label='Equal Speed')
        ax.set_xlim(limit)
        ax.set_ylim(limit)
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        ax.fill_between(limit, limit[0], limit, color='green', alpha=0.05) 
        
        ax.set_title(f"{label_name} vs Baseline\n(Model Faster: {model_wins}, Baseline Faster: {base_wins})", fontweight='bold')
        ax.set_xlabel("Baseline Nodes (Log)")
        ax.set_ylabel(f"{label_name} Nodes (Log)")
        
        ax.text(limit[1]*0.9, limit[0]*1.5, f"{label_name} Faster", 
                 fontsize=10, color='darkgreen', ha='right', va='bottom', fontweight='bold')
        ax.text(limit[0]*1.2, limit[1]*0.8, "Baseline Faster", 
                 fontsize=10, color='gray', ha='left', va='top')

    # pre transformer vs baseline
    plot_scatter(axs[1, 0], 'Nodes_Pre', 'Status_Pre', c_pre, "Pre-Train")

    # current transformer vs baseline
    plot_scatter(axs[1, 1], 'Nodes_Pred', 'Status_Pred', c_curr, "Current Transformer")

    print(f"\nGenerating plots to {OUTPUT_IMG}...")
    plt.savefig(OUTPUT_IMG, dpi=150)
    print("Done! Check the image.")

if __name__ == "__main__":
    analyze()