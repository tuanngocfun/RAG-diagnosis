#!/usr/bin/env python3
"""
Retrieval Experiments Visualization Script
Generates comprehensive charts for retrieval system performance metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(csv_path):
    """Load and prepare retrieval experiments data."""
    df = pd.read_csv(csv_path)
    
    # Create meaningful, unique system names
    df['experiment_clean'] = df.apply(create_unique_retrieval_name, axis=1)
    
    # Remove rows with all zero values (like standalone Gemini)
    df = df[(df[['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']] != 0).any(axis=1)]
    
    return df

def create_unique_retrieval_name(row):
    """Create unique, meaningful names for retrieval systems."""
    name = row['experiment']
    eval_type = row['type'] if 'type' in row else 'unknown'
    
    # Define mapping for cleaner, unique names
    name_mapping = {
        'Bi-encoder + MedCPT reranker': 'BiEnc+MedCPT',
        'Cross-encoder + MedCPT pool6 top6': 'CrossEnc+MedCPT-P6',
        'Cross-encoder + MedCPT reranker': 'CrossEnc+MedCPT',
        'Cross-encoder + MedCPT reranker (patched final)': 'CrossEnc+MedCPT-Patched',
        'Offline eval: Gemini2.5 + BGE reranker': 'Gemini25+BGE-Offline',
        'Offline eval: MedGemma4b + MedCPT reranker': 'MedGemma4b+MedCPT-Offline',
        'Standalone Gemini2.5 (no retriever)': 'Gemini25-Standalone',
        'Standalone MedGemma4b (no retriever)': 'MedGemma4b-Standalone'
    }
    
    # Use mapping if available
    if name in name_mapping:
        base_name = name_mapping[name]
    else:
        # Create name from components
        clean_name = name.lower()
        components = []
        
        # Encoder type
        if 'bi-encoder' in clean_name:
            components.append('BiEnc')
        elif 'cross-encoder' in clean_name:
            components.append('CrossEnc')
        
        # Generator/Model
        if 'gemini2.5' in clean_name or 'gemini25' in clean_name:
            components.append('Gemini25')
        elif 'medgemma4b' in clean_name:
            components.append('MedGemma4b')
        
        # Reranker
        if 'medcpt' in clean_name:
            if 'pool6' in clean_name:
                components.append('MedCPT-P6')
            else:
                components.append('MedCPT')
        elif 'bge' in clean_name:
            components.append('BGE')
        
        # Special attributes
        if 'offline' in clean_name:
            components.append('Offline')
        if 'patched' in clean_name:
            components.append('Patched')
        if 'final' in clean_name:
            components.append('Final')
        if 'standalone' in clean_name:
            components.append('Standalone')
        
        base_name = '+'.join(components) if components else name[:15]
    
    # Add evaluation type suffix for clarity
    if eval_type == 'inline_eval':
        return f'{base_name}'
    elif eval_type == 'posthoc_eval':
        return f'{base_name}' if 'Offline' in base_name else f'{base_name}-Posthoc'
    else:
        return base_name

def create_retrieval_metrics_chart(df, output_dir):
    """Create comprehensive retrieval metrics comparison."""
    metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']
    metric_names = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.ravel()
    
    colors = sns.color_palette("husl", len(df))
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # Sort by metric value for better visualization
        df_sorted = df.sort_values(metric, ascending=True)
        
        bars = ax.barh(range(len(df_sorted)), df_sorted[metric], color=colors)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['experiment_clean'], fontsize=9, ha='right')
        ax.set_xlabel(f'{name} Score', fontsize=12)
        ax.set_title(f'{name} Performance Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Adjust layout for better label visibility
        ax.margins(y=0.01)
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, df_sorted[metric])):
            ax.text(value + 0.01, j, f'{value:.3f}', 
                   va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'retrieval_performance_metrics.pdf', bbox_inches='tight')
    plt.show()

def create_success_rate_analysis(df, output_dir):
    """Create analysis of query success rates and coverage."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Success Rate Analysis
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('successful_queries', ascending=True)
    bars1 = ax1.barh(range(len(df_sorted)), 
                     df_sorted['successful_queries'] / df_sorted['total_queries'] * 100,
                     color='skyblue')
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['experiment_clean'], fontsize=9, ha='right')
    ax1.set_xlabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Query Success Rate', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, total, success) in enumerate(zip(bars1, df_sorted['total_queries'], df_sorted['successful_queries'])):
        pct = success / total * 100
        ax1.text(pct + 1, i, f'{pct:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    # Gold Pages Coverage
    ax2 = axes[0, 1]
    df_filtered = df[df['queries_with_gold_pages'] > 0].sort_values('queries_with_gold_pages', ascending=True)
    bars2 = ax2.barh(range(len(df_filtered)), df_filtered['queries_with_gold_pages'], color='lightgreen')
    ax2.set_yticks(range(len(df_filtered)))
    ax2.set_yticklabels(df_filtered['experiment_clean'], fontsize=9, ha='right')
    ax2.set_xlabel('Queries with Gold Pages', fontsize=12)
    ax2.set_title('Gold Pages Coverage', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars2, df_filtered['queries_with_gold_pages'])):
        ax2.text(value + 10, i, f'{int(value)}', va='center', fontsize=9, fontweight='bold')
    
    # Text Coverage Analysis
    ax3 = axes[1, 0]
    df_coverage = df[df['text_coverage@k'] > 0].sort_values('text_coverage@k', ascending=True)
    bars3 = ax3.barh(range(len(df_coverage)), df_coverage['text_coverage@k'], color='coral')
    ax3.set_yticks(range(len(df_coverage)))
    ax3.set_yticklabels(df_coverage['experiment_clean'], fontsize=9, ha='right')
    ax3.set_xlabel('Text Coverage@k', fontsize=12)
    ax3.set_title('Text Coverage Performance', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars3, df_coverage['text_coverage@k'])):
        ax3.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # Duplicate Ratio Analysis
    ax4 = axes[1, 1]
    df_dup = df[df['duplicate_ratio@k'] >= 0].sort_values('duplicate_ratio@k', ascending=False)
    bars4 = ax4.barh(range(len(df_dup)), df_dup['duplicate_ratio@k'] * 100, color='orange')
    ax4.set_yticks(range(len(df_dup)))
    ax4.set_yticklabels(df_dup['experiment_clean'], fontsize=9, ha='right')
    ax4.set_xlabel('Duplicate Ratio (%)', fontsize=12)
    ax4.set_title('Retrieval Duplicate Rate', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, value) in enumerate(zip(bars4, df_dup['duplicate_ratio@k'])):
        ax4.text(value * 100 + 0.5, i, f'{value*100:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_success_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'retrieval_success_analysis.pdf', bbox_inches='tight')
    plt.show()

def create_retrieval_radar_chart(df, output_dir):
    """Create radar chart for retrieval system comparison."""
    metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k', 'doc_recall@k', 'text_coverage@k']
    metric_labels = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k', 'Doc Recall@k', 'Text Coverage@k']
    
    # Filter out rows with all zeros and normalize metrics
    df_filtered = df[(df[metrics] != 0).any(axis=1)].copy()
    df_norm = df_filtered.copy()
    
    for metric in metrics:
        max_val = df_filtered[metric].max()
        if max_val > 0:
            df_norm[metric] = df_filtered[metric] / max_val
        else:
            df_norm[metric] = 0
    
    # Set up radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(projection='polar'))
    
    colors = sns.color_palette("husl", len(df_norm))
    
    for i, (_, row) in enumerate(df_norm.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['experiment_clean'], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Retrieval Systems Performance Radar Chart\n(Normalized Metrics)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'retrieval_radar_chart.pdf', bbox_inches='tight')
    plt.show()

def create_evaluation_type_comparison(df, output_dir):
    """Compare performance between inline_eval and posthoc_eval."""
    if 'type' not in df.columns:
        print("⚠️ No 'type' column found for evaluation type comparison")
        return
    
    eval_types = df['type'].unique()
    if len(eval_types) <= 1:
        print("⚠️ Only one evaluation type found, skipping comparison")
        return
    
    metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create boxplot comparing evaluation types
        sns.boxplot(data=df, x='type', y=metric, ax=ax)
        sns.stripplot(data=df, x='type', y=metric, ax=ax, color='red', alpha=0.7)
        
        ax.set_title(f'{metric} by Evaluation Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Evaluation Type', fontsize=12)
        ax.set_ylabel(f'{metric} Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add mean values as text
        for j, eval_type in enumerate(eval_types):
            type_data = df[df['type'] == eval_type][metric]
            if len(type_data) > 0:
                mean_val = type_data.mean()
                ax.text(j, mean_val, f'μ={mean_val:.3f}', 
                       ha='center', va='bottom', fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_eval_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'retrieval_eval_type_comparison.pdf', bbox_inches='tight')
    plt.show()

def create_correlation_heatmap(df, output_dir):
    """Create correlation heatmap for retrieval metrics."""
    metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k', 'doc_recall@k', 'text_coverage@k']
    
    # Filter numeric data
    df_numeric = df[metrics].select_dtypes(include=[np.number])
    
    if df_numeric.empty:
        print("⚠️ No numeric data found for correlation analysis")
        return
    
    corr_matrix = df_numeric.corr()
    
    plt.figure(figsize=(12, 10))
    
    # Create correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.3f',
                cmap='coolwarm',
                center=0,
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Retrieval Metrics Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'retrieval_correlation_matrix.pdf', bbox_inches='tight')
    plt.show()

def create_performance_summary_table(df, output_dir):
    """Create summary table of best performing systems."""
    metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k', 'doc_recall@k', 'text_coverage@k']
    
    # Find best system for each metric
    best_systems = {}
    for metric in metrics:
        if metric in df.columns and df[metric].max() > 0:
            best_idx = df[metric].idxmax()
            best_systems[metric] = {
                'system': df.loc[best_idx, 'experiment_clean'],
                'score': df.loc[best_idx, metric],
                'type': df.loc[best_idx, 'type'] if 'type' in df.columns else 'N/A'
            }
    
    # Create summary dataframe
    summary_data = []
    for metric, info in best_systems.items():
        summary_data.append({
            'Metric': metric,
            'Best System': info['system'],
            'Score': f"{info['score']:.3f}",
            'Eval Type': info['type']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_df.to_csv(output_dir / 'retrieval_best_systems.csv', index=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(len(summary_df)):
        for j in range(len(summary_df.columns)):
            if j % 2 == 0:
                table[(i+1, j)].set_facecolor('#E8F5E8')
    
    plt.title('Best Performing Retrieval Systems by Metric', 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'retrieval_best_systems_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'retrieval_best_systems_table.pdf', bbox_inches='tight')
    plt.show()
    
    return summary_df

def main():
    """Main function to generate all retrieval evaluation visualizations."""
    # Setup paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'Final_Retrieval_Experiments_Comparison.csv'
    output_dir = script_dir
    
    print("🔄 Loading retrieval experiments data...")
    df = load_and_prepare_data(csv_path)
    
    print(f"✅ Loaded {len(df)} retrieval systems for analysis")
    print(f"📊 Generating visualizations in: {output_dir}")
    
    # Generate all visualizations
    print("📈 Creating retrieval metrics chart...")
    create_retrieval_metrics_chart(df, output_dir)
    
    print("📊 Creating success rate analysis...")
    create_success_rate_analysis(df, output_dir)
    
    print("🎯 Creating radar chart...")
    create_retrieval_radar_chart(df, output_dir)
    
    print("🔍 Creating evaluation type comparison...")
    create_evaluation_type_comparison(df, output_dir)
    
    print("🔗 Creating correlation heatmap...")
    create_correlation_heatmap(df, output_dir)
    
    print("🏆 Creating performance summary table...")
    summary_df = create_performance_summary_table(df, output_dir)
    
    print("\n" + "="*70)
    print("📊 RETRIEVAL EXPERIMENTS SUMMARY")
    print("="*70)
    print(f"Number of systems evaluated: {len(df)}")
    
    if 'type' in df.columns:
        type_counts = df['type'].value_counts()
        print("Evaluation types:")
        for eval_type, count in type_counts.items():
            print(f"  {eval_type}: {count} systems")
    
    print("\nBest performing systems:")
    metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']
    for metric in metrics:
        if metric in df.columns and df[metric].max() > 0:
            best_idx = df[metric].idxmax()
            best_system = df.loc[best_idx, 'experiment_clean']
            best_score = df.loc[best_idx, metric]
            print(f"  {metric}: {best_system} ({best_score:.3f})")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()