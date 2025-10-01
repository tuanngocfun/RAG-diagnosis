#!/usr/bin/env python3
"""
RAG Evaluation Visualization Script
Generates comprehensive charts for RAG system performance metrics.
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
    """Load and prepare RAG evaluation data."""
    df = pd.read_csv(csv_path)
    
    # Create meaningful, unique system names
    df['run_clean'] = df.apply(create_unique_rag_name, axis=1)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def create_unique_rag_name(row):
    """Create unique, meaningful names for RAG systems."""
    name = row['run']
    path = row['path'] if 'path' in row else ''
    
    # Define mapping for cleaner names
    name_mapping = {
        'text_size12k, token1024 (missing)': 'Gemini25-12k-1024tok',
        'gem25-gemini-rpm-5 (missing)': 'Gemini25-Base',
        'gemimi_rerank_rag_test (all)': 'Gemini-Rerank-RAG',
        'RTX3090 cross-enc MedCPT (final, all)': 'CrossEnc-MedCPT-Final',
        'bi-encoder reranker (all)': 'BiEncoder-Rerank',
        'cross-encoder reranker (all)': 'CrossEnc-Rerank',
        'cross-encoder MedCPT pool6 top6 (all)': 'CrossEnc-MedCPT-Pool6',
        'MedGemma4B standalone (all)': 'MedGemma4B-Standalone'
    }
    
    # Use mapping if available, otherwise create from components
    if name in name_mapping:
        return name_mapping[name]
    
    # Extract key components for unique naming
    clean_name = name.lower()
    components = []
    
    # Generator type
    if 'gem25' in clean_name or 'gemini' in clean_name:
        components.append('Gemini25')
    elif 'medgemma' in clean_name:
        components.append('MedGemma')
    
    # Retrieval method
    if 'cross-enc' in clean_name or 'crossencoder' in clean_name:
        components.append('CrossEnc')
    elif 'bi-encoder' in clean_name or 'biencoder' in clean_name:
        components.append('BiEnc')
    
    # Reranker type
    if 'medcpt' in clean_name:
        if 'pool6' in clean_name:
            components.append('MedCPT-P6')
        else:
            components.append('MedCPT')
    elif 'rerank' in clean_name:
        components.append('Rerank')
    
    # Special attributes
    if 'final' in clean_name:
        components.append('Final')
    if '12k' in clean_name:
        components.append('12k')
    if '1024' in clean_name:
        components.append('1024tok')
    
    # Data strategy
    if 'missing' in clean_name:
        components.append('Missing')
    elif 'all' in clean_name:
        components.append('All')
    
    if components:
        return '-'.join(components)
    else:
        # Fallback: clean the original name
        clean = name.replace('(', '').replace(')', '').replace(',', '')
        return clean[:20]  # Limit length

def create_performance_metrics_chart(df, output_dir):
    """Create comprehensive performance metrics comparison."""
    metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    metric_names = ['F1 Score', 'Faithfulness', 'Correctness', 'Completeness']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.ravel()
    
    colors = sns.color_palette("husl", len(df))
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # Sort by metric value for better visualization
        df_sorted = df.sort_values(metric, ascending=True)
        
        bars = ax.barh(range(len(df_sorted)), df_sorted[metric], color=colors)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['run_clean'], fontsize=9, ha='right')
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
    plt.savefig(output_dir / 'rag_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'rag_performance_metrics.pdf', bbox_inches='tight')
    plt.show()

def create_radar_chart(df, output_dir):
    """Create radar chart for multi-dimensional performance comparison."""
    metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    metric_labels = ['F1 Score', 'Faithfulness', 'Correctness', 'Completeness']
    
    # Normalize metrics to 0-1 scale for better radar visualization
    df_norm = df.copy()
    for metric in metrics:
        df_norm[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
    
    # Set up radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(projection='polar'))
    
    colors = sns.color_palette("husl", len(df))
    
    for i, (_, row) in enumerate(df_norm.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['run_clean'], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('RAG Systems Performance Radar Chart\n(Normalized Metrics)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rag_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'rag_radar_chart.pdf', bbox_inches='tight')
    plt.show()

def create_ranking_heatmap(df, output_dir):
    """Create heatmap showing ranking across different metrics."""
    ranking_cols = ['rank_F1', 'rank_faithfulness', 'rank_correctness', 'rank_completeness']
    ranking_labels = ['F1 Rank', 'Faithfulness Rank', 'Correctness Rank', 'Completeness Rank']
    
    # Prepare ranking matrix
    ranking_matrix = df[ranking_cols].values
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with reversed colormap (lower rank = better = darker)
    sns.heatmap(ranking_matrix, 
                annot=True, 
                fmt='d',
                xticklabels=ranking_labels,
                yticklabels=df['run_clean'],
                cmap='RdYlGn_r',  # Reversed: red=bad rank, green=good rank
                cbar_kws={'label': 'Rank (1=Best)'},
                square=False)
    
    plt.title('RAG Systems Ranking Heatmap\n(1=Best Performance)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Evaluation Metrics', fontsize=12)
    plt.ylabel('RAG Systems', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rag_ranking_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'rag_ranking_heatmap.pdf', bbox_inches='tight')
    plt.show()

def create_timeline_chart(df, output_dir):
    """Create timeline showing performance evolution over time."""
    df_sorted = df.sort_values('timestamp')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    
    metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    metric_names = ['F1 Score', 'Faithfulness', 'Correctness', 'Completeness']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
        ax = axes[i]
        
        ax.plot(df_sorted['timestamp'], df_sorted[metric], 
                'o-', linewidth=2, markersize=8, color=color)
        
        # Add labels for each point
        for j, (ts, value, run_name) in enumerate(zip(df_sorted['timestamp'], 
                                                     df_sorted[metric], 
                                                     df_sorted['run_clean'])):
            ax.annotate(run_name, (ts, value), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, rotation=45, alpha=0.8, ha='left')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{name} Score', fontsize=12)
        ax.set_title(f'{name} Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rag_timeline_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'rag_timeline_performance.pdf', bbox_inches='tight')
    plt.show()

def create_correlation_matrix(df, output_dir):
    """Create correlation matrix for RAG metrics."""
    metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    metric_names = ['F1 Score', 'Faithfulness', 'Correctness', 'Completeness']
    
    corr_matrix = df[metrics].corr()
    
    plt.figure(figsize=(10, 8))
    
    # Create correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.3f',
                xticklabels=metric_names,
                yticklabels=metric_names,
                cmap='coolwarm',
                center=0,
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('RAG Metrics Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rag_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'rag_correlation_matrix.pdf', bbox_inches='tight')
    plt.show()

def generate_summary_stats(df, output_dir):
    """Generate and save summary statistics."""
    metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    
    summary_stats = df[metrics].describe()
    
    # Save to CSV
    summary_stats.to_csv(output_dir / 'rag_summary_statistics.csv')
    
    # Create summary table visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_stats.round(4).values,
                    rowLabels=summary_stats.index,
                    colLabels=['F1 Score', 'Faithfulness', 'Correctness', 'Completeness'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(summary_stats.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(len(summary_stats.index)):
        table[(i+1, -1)].set_facecolor('#E8F5E8')
    
    plt.title('RAG Evaluation Metrics - Summary Statistics', 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'rag_summary_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'rag_summary_table.pdf', bbox_inches='tight')
    plt.show()
    
    return summary_stats

def main():
    """Main function to generate all RAG evaluation visualizations."""
    # Setup paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'Updated_RAG_experiment_comparison_with_MedGemma4B_standalone.csv'
    output_dir = script_dir
    
    print("🔄 Loading RAG evaluation data...")
    df = load_and_prepare_data(csv_path)
    
    print(f"✅ Loaded {len(df)} RAG systems for analysis")
    print(f"📊 Generating visualizations in: {output_dir}")
    
    # Generate all visualizations
    print("📈 Creating performance metrics chart...")
    create_performance_metrics_chart(df, output_dir)
    
    print("🎯 Creating radar chart...")
    create_radar_chart(df, output_dir)
    
    print("🔥 Creating ranking heatmap...")
    create_ranking_heatmap(df, output_dir)
    
    print("⏱️ Creating timeline chart...")
    create_timeline_chart(df, output_dir)
    
    print("🔗 Creating correlation matrix...")
    create_correlation_matrix(df, output_dir)
    
    print("📋 Generating summary statistics...")
    summary_stats = generate_summary_stats(df, output_dir)
    
    print("\n" + "="*60)
    print("📊 RAG EVALUATION SUMMARY")
    print("="*60)
    print(f"Number of systems evaluated: {len(df)}")
    print(f"Date range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
    print("\nTop performing systems:")
    for metric in ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']:
        best_idx = df[metric].idxmax()
        best_system = df.loc[best_idx, 'run_clean']
        best_score = df.loc[best_idx, metric]
        print(f"  {metric.replace('avg_', '').title()}: {best_system} ({best_score:.3f})")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()