#!/usr/bin/env python3
"""
Generate comprehensive visualizations from evaluation results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10

def load_data():
    """Load the evaluation results CSV."""
    csv_path = Path("out/complete_evaluation_results.csv")
    df = pd.read_csv(csv_path)
    return df

def create_model_labels():
    """Create readable model labels."""
    return {
        "Llama-3.1-8B-Instruct": "Llama 3.1-8B (Teacher)",
        "Llama-3.2-1B-Instruct": "Llama 3.2-1B (Base)",
        "llama_instruct_from_llama_merged": "Llama 3.2-1B (Student)",
        "llama_from_qwen_merged": "Llama 3.2-1B (Cross-Family)",
        "Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math-7B (Teacher)",
        "Qwen2.5-1.5B": "Qwen2.5-1.5B (Base)",
        "qwen_from_qwen_merged": "Qwen2.5-1.5B (Student)",
        "qwen_from_llama_merged": "Qwen2.5-1.5B (Cross-Family)",
    }

def plot_1_model_comparison(df, output_dir):
    """Plot 1: Model Performance Comparison on MATH-500 at full context."""
    print("[INFO] Creating Plot 1: Model Performance Comparison...")
    
    # Filter for MATH-500 at prefix_ratio=1.0
    data = df[(df['dataset'] == 'math500') & (df['prefix_ratio'] == 1.0)].copy()
    
    # Create readable labels
    model_labels = create_model_labels()
    data['model_label'] = data['model'].map(model_labels)
    
    # Define colors: Teachers=blue, Base=purple, Students=orange, Cross-family=green
    colors = {
        "Llama 3.1-8B (Teacher)": "#2E86AB",
        "Llama 3.2-1B (Base)": "#A23B72",
        "Llama 3.2-1B (Student)": "#F18F01",
        "Llama 3.2-1B (Cross-Family)": "#06A77D",
        "Qwen2.5-Math-7B (Teacher)": "#2E86AB",
        "Qwen2.5-1.5B (Base)": "#A23B72",
        "Qwen2.5-1.5B (Student)": "#C73E1D",
        "Qwen2.5-1.5B (Cross-Family)": "#06A77D",
    }
    
    # Sort by accuracy
    data = data.sort_values('accuracy', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.barh(data['model_label'], data['accuracy'] * 100, 
                   color=[colors[m] for m in data['model_label']],
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add value labels
    for i, (idx, row) in enumerate(data.iterrows()):
        ax.text(row['accuracy'] * 100 + 1, i, 
                f"{row['accuracy']*100:.1f}% ({row['correct']}/{row['total']})",
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance on MATH-500 (Full Context)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 80)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot1_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ✓ Saved plot1_model_comparison.png")

def plot_2_memorization_analysis(df, output_dir):
    """Plot 2: Memorization Analysis - Accuracy vs Prefix Ratio."""
    print("[INFO] Creating Plot 2: Memorization Analysis...")
    
    # Filter for MATH-500 only
    data = df[df['dataset'] == 'math500'].copy()
    
    model_labels = create_model_labels()
    data['model_label'] = data['model'].map(model_labels)
    
    # Convert prefix_ratio to float for sorting
    data['prefix_ratio_float'] = data['prefix_ratio'].astype(float)
    data = data.sort_values('prefix_ratio_float')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define line styles
    styles = {
        "Llama 3.1-8B (Teacher)": {'color': '#2E86AB', 'linestyle': '-', 'marker': 'o', 'linewidth': 2.5},
        "Llama 3.2-1B (Base)": {'color': '#A23B72', 'linestyle': '--', 'marker': 's', 'linewidth': 2},
        "Llama 3.2-1B (Student)": {'color': '#F18F01', 'linestyle': '--', 'marker': '^', 'linewidth': 2},
        "Llama 3.2-1B (Cross-Family)": {'color': '#06A77D', 'linestyle': '-.', 'marker': 'D', 'linewidth': 2},
        "Qwen2.5-Math-7B (Teacher)": {'color': '#2E86AB', 'linestyle': '-', 'marker': 'o', 'linewidth': 2.5},
        "Qwen2.5-1.5B (Base)": {'color': '#D96C06', 'linestyle': '--', 'marker': 's', 'linewidth': 2},
        "Qwen2.5-1.5B (Student)": {'color': '#C73E1D', 'linestyle': '--', 'marker': '^', 'linewidth': 2},
        "Qwen2.5-1.5B (Cross-Family)": {'color': '#06A77D', 'linestyle': '-.', 'marker': 'D', 'linewidth': 2},
    }
    
    for model_label in data['model_label'].unique():
        model_data = data[data['model_label'] == model_label]
        style = styles[model_label]
        ax.plot(model_data['prefix_ratio_float'], model_data['accuracy'] * 100,
                label=model_label, **style, markersize=8)
    
    ax.set_xlabel('Prefix Ratio (Portion of Question Shown)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Memorization Analysis: How Accuracy Changes with Question Truncation (MATH-500)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0.4\n(40%)', '0.6\n(60%)', '0.8\n(80%)', '1.0\n(100%)'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot2_memorization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ✓ Saved plot2_memorization_analysis.png")

def plot_3_teacher_student_comparison(df, output_dir):
    """Plot 3: Teacher-Student Knowledge Transfer."""
    print("[INFO] Creating Plot 3: Teacher-Student Knowledge Transfer...")
    
    # Filter for MATH-500 at prefix_ratio=1.0
    data = df[(df['dataset'] == 'math500') & (df['prefix_ratio'] == 1.0)].copy()
    
    model_labels = create_model_labels()
    data['model_label'] = data['model'].map(model_labels)
    
    # Define groups
    llama_models = [
        "Llama 3.1-8B (Teacher)",
        "Llama 3.2-1B (Base)",
        "Llama 3.2-1B (Student)"
    ]
    
    qwen_models = [
        "Qwen2.5-Math-7B (Teacher)",
        "Qwen2.5-1.5B (Base)",
        "Qwen2.5-1.5B (Student)"
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Llama family
    llama_data = data[data['model_label'].isin(llama_models)]
    llama_data = llama_data.set_index('model_label').reindex(llama_models)
    
    bars1 = ax1.bar(range(len(llama_models)), llama_data['accuracy'] * 100,
                    color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(llama_models)))
    ax1.set_xticklabels(['Teacher\n(8B)', 'Base\n(1B)', 'Student\n(1B)'], fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Llama Family', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 80)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars1, llama_data['accuracy'])):
        ax1.text(bar.get_x() + bar.get_width()/2, acc * 100 + 2,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Clean up spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Qwen family
    qwen_data = data[data['model_label'].isin(qwen_models)]
    qwen_data = qwen_data.set_index('model_label').reindex(qwen_models)
    
    bars2 = ax2.bar(range(len(qwen_models)), qwen_data['accuracy'] * 100,
                    color=['#06A77D', '#D96C06', '#C73E1D'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(qwen_models)))
    ax2.set_xticklabels(['Teacher\n(7B)', 'Base\n(1.5B)', 'Student\n(1.5B)'], fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Qwen Family', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 80)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars2, qwen_data['accuracy'])):
        ax2.text(bar.get_x() + bar.get_width()/2, acc * 100 + 2,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Clean up spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle('Knowledge Distillation: Teacher → Student Transfer (MATH-500)',
                 fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot3_teacher_student_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ✓ Saved plot3_teacher_student_comparison.png")

def plot_4_answer_format_quality(df, output_dir):
    """Plot 4: Answer Format Quality (boxed vs no boxed)."""
    print("[INFO] Creating Plot 4: Answer Format Quality...")
    
    # Filter for MATH-500 at prefix_ratio=1.0
    data = df[(df['dataset'] == 'math500') & (df['prefix_ratio'] == 1.0)].copy()
    
    model_labels = create_model_labels()
    data['model_label'] = data['model'].map(model_labels)
    
    # Calculate percentages
    data['valid_boxed_pct'] = (1 - data['no_boxed_ratio']) * 100
    data['no_boxed_pct'] = data['no_boxed_ratio'] * 100
    
    # Sort by valid boxed percentage
    data = data.sort_values('valid_boxed_pct')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create stacked bar chart
    bars1 = ax.barh(data['model_label'], data['valid_boxed_pct'], 
                    label='Valid \\boxed{} Answer', color='#06A77D', alpha=0.85,
                    edgecolor='black', linewidth=1.5)
    bars2 = ax.barh(data['model_label'], data['no_boxed_pct'], 
                    left=data['valid_boxed_pct'],
                    label='No \\boxed{} Answer', color='#C73E1D', alpha=0.85,
                    edgecolor='black', linewidth=1.5)
    
    # Add percentage labels
    for i, (idx, row) in enumerate(data.iterrows()):
        # Valid boxed label
        if row['valid_boxed_pct'] > 5:
            ax.text(row['valid_boxed_pct']/2, i, 
                    f"{row['valid_boxed_pct']:.1f}%",
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        # No boxed label
        if row['no_boxed_pct'] > 5:
            ax.text(row['valid_boxed_pct'] + row['no_boxed_pct']/2, i,
                    f"{row['no_boxed_pct']:.1f}%",
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    ax.set_xlabel('Percentage of Responses (%)', fontsize=12, fontweight='bold')
    ax.set_title('Answer Format Quality: Valid vs Missing \\boxed{} Answers (MATH-500)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot4_answer_format_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ✓ Saved plot4_answer_format_quality.png")

def plot_5_rouge_memorization(df, output_dir):
    """Plot 5: ROUGE-L Memorization Pattern."""
    print("[INFO] Creating Plot 5: ROUGE-L Memorization Pattern...")
    
    # Filter for MATH-500 only (exclude prefix_ratio=1.0 as ROUGE-L is always 0)
    data = df[(df['dataset'] == 'math500') & (df['prefix_ratio'] != 1.0)].copy()
    
    model_labels = create_model_labels()
    data['model_label'] = data['model'].map(model_labels)
    
    # Convert prefix_ratio to float
    data['prefix_ratio_float'] = data['prefix_ratio'].astype(float)
    data = data.sort_values('prefix_ratio_float')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define line styles
    styles = {
        "Llama 3.1-8B (Teacher)": {'color': '#2E86AB', 'linestyle': '-', 'marker': 'o', 'linewidth': 2.5},
        "Llama 3.2-1B (Base)": {'color': '#A23B72', 'linestyle': '--', 'marker': 's', 'linewidth': 2},
        "Llama 3.2-1B (Student)": {'color': '#F18F01', 'linestyle': '--', 'marker': '^', 'linewidth': 2},
        "Llama 3.2-1B (Cross-Family)": {'color': '#06A77D', 'linestyle': '-.', 'marker': 'D', 'linewidth': 2},
        "Qwen2.5-Math-7B (Teacher)": {'color': '#2E86AB', 'linestyle': '-', 'marker': 'o', 'linewidth': 2.5},
        "Qwen2.5-1.5B (Base)": {'color': '#D96C06', 'linestyle': '--', 'marker': 's', 'linewidth': 2},
        "Qwen2.5-1.5B (Student)": {'color': '#C73E1D', 'linestyle': '--', 'marker': '^', 'linewidth': 2},
        "Qwen2.5-1.5B (Cross-Family)": {'color': '#06A77D', 'linestyle': '-.', 'marker': 'D', 'linewidth': 2},
    }
    
    for model_label in data['model_label'].unique():
        model_data = data[data['model_label'] == model_label]
        style = styles[model_label]
        ax.plot(model_data['prefix_ratio_float'], model_data['avg_rougeL_f'],
                label=model_label, **style, markersize=8)
    
    ax.set_xlabel('Prefix Ratio (Portion of Question Shown)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROUGE-L F-score', fontsize=12, fontweight='bold')
    ax.set_title('Text-Level Memorization: ROUGE-L vs Question Truncation (MATH-500)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xticks([0.4, 0.6, 0.8])
    ax.set_xticklabels(['0.4\n(40%)', '0.6\n(60%)', '0.8\n(80%)'])
    
    # Add annotation
    ax.text(0.95, 0.05, 'Higher ROUGE-L = More verbatim memorization',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot5_rouge_memorization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ✓ Saved plot5_rouge_memorization.png")

def plot_6_dataset_difficulty_heatmap(df, output_dir):
    """Plot 6: Dataset Difficulty Comparison Heatmap."""
    print("[INFO] Creating Plot 6: Dataset Difficulty Heatmap...")
    
    # Filter for prefix_ratio=1.0
    data = df[df['prefix_ratio'] == 1.0].copy()
    
    model_labels = create_model_labels()
    data['model_label'] = data['model'].map(model_labels)
    
    # Pivot for heatmap
    pivot_data = data.pivot(index='model_label', columns='dataset', values='accuracy')
    pivot_data = pivot_data * 100  # Convert to percentage
    
    # Rename columns first
    pivot_data.columns = ['AIME-2025' if c == 'aime2025' else 'MATH-500' for c in pivot_data.columns]
    
    # Reorder rows
    row_order = [
        "Qwen2.5-Math-7B (Teacher)",
        "Qwen2.5-1.5B (Student)",
        "Qwen2.5-1.5B (Base)",
        "Llama 3.1-8B (Teacher)",
        "Llama 3.2-1B (Student)",
        "Llama 3.2-1B (Base)",
    ]
    pivot_data = pivot_data.reindex(row_order)
    
    # Reorder columns to have MATH-500 first
    if 'MATH-500' in pivot_data.columns and 'AIME-2025' in pivot_data.columns:
        pivot_data = pivot_data[['MATH-500', 'AIME-2025']]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=0, vmax=75, cbar_kws={'label': 'Accuracy (%)'},
                linewidths=2, linecolor='white', ax=ax,
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_title('Dataset Difficulty Comparison: Model Performance Heatmap',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot6_dataset_difficulty_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ✓ Saved plot6_dataset_difficulty_heatmap.png")

def main():
    print("="*80)
    print("EVALUATION RESULTS VISUALIZATION")
    print("="*80)
    print()
    
    # Load data
    print("[INFO] Loading data from out/complete_evaluation_results.csv...")
    df = load_data()
    print(f"[INFO] Loaded {len(df)} rows")
    print()
    
    # Create output directory
    output_dir = Path("out/visualizations")
    output_dir.mkdir(exist_ok=True)
    print(f"[INFO] Saving visualizations to {output_dir}/")
    print()
    
    # Generate all plots
    plot_1_model_comparison(df, output_dir)
    plot_2_memorization_analysis(df, output_dir)
    plot_3_teacher_student_comparison(df, output_dir)
    plot_4_answer_format_quality(df, output_dir)
    plot_5_rouge_memorization(df, output_dir)
    plot_6_dataset_difficulty_heatmap(df, output_dir)
    
    print()
    print("="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"6 plots saved to: {output_dir}/")
    print()

if __name__ == "__main__":
    main()

