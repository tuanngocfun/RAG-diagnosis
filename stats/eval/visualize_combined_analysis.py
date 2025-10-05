#!/usr/bin/env python3
"""
Combined Analysis Dashboard Script
Generates unified comparison visualizations combining both RAG and retrieval datasets.
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

def create_unique_rag_name(row):
    """Create unique, meaningful names for RAG systems."""
    name = row['run']
    
    name_mapping = {
        'text_size12k, token1024 (missing)': 'Gemini 2.5 Pro: 12k-1024tok',
        'gem25-gemini-rpm-5 (missing)': 'Gemini 2.5 Pro: Base',
        'gemimi_rerank_rag_test (all)': 'Gemini 2.5 Pro: Rerank-RAG',
        'RTX3090 cross-enc MedCPT (final, all)': 'MedGemma-4B: CrossEnc-MedCPT',
        'bi-encoder reranker (all)': 'MedGemma-4B: BiEncoder',
        'cross-encoder reranker (all)': 'MedGemma-4B: CrossEnc',
        'cross-encoder MedCPT pool6 top6 (all)': 'MedGemma-4B: CrossEnc-Pool6',
        'MedGemma4B standalone (all)': 'MedGemma-4B: Standalone'
    }
    
    return name_mapping.get(name, name[:20])

def create_unique_retrieval_name(row):
    """Create unique, meaningful names for retrieval systems."""
    name = row['experiment']
    
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
    
    return name_mapping.get(name, name[:20])

def load_both_datasets(script_dir):
    """Load and prepare both RAG and retrieval datasets."""
    rag_path = script_dir / 'Updated_RAG_experiment_comparison_with_MedGemma4B_standalone.csv'
    retrieval_path = script_dir / 'Final_Retrieval_Experiments_Comparison.csv'
    
    # Load RAG data
    rag_df = pd.read_csv(rag_path)
    rag_df['run_clean'] = rag_df.apply(create_unique_rag_name, axis=1)
    rag_df['dataset'] = 'RAG_Evaluation'
    
    # Load Retrieval data
    retrieval_df = pd.read_csv(retrieval_path)
    retrieval_df['experiment_clean'] = retrieval_df.apply(create_unique_retrieval_name, axis=1)
    retrieval_df['dataset'] = 'Retrieval_Experiments'
    
    # Remove zero-performance systems from retrieval
    retrieval_df = retrieval_df[(retrieval_df[['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']] != 0).any(axis=1)]
    
    return rag_df, retrieval_df

def create_system_overview(rag_df, retrieval_df, output_dir):
    """Create overview of all systems across both datasets."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RAG Systems Performance Overview
    ax1 = axes[0, 0]
    rag_metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    rag_means = [rag_df[metric].mean() for metric in rag_metrics]
    rag_stds = [rag_df[metric].std() for metric in rag_metrics]
    
    x_pos = np.arange(len(rag_metrics))
    bars1 = ax1.bar(x_pos, rag_means, yerr=rag_stds, capsize=5, 
                    color='skyblue', alpha=0.7, label='Mean ± Std')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['F1', 'Faithfulness', 'Correctness', 'Completeness'], rotation=45)
    ax1.set_title('RAG Systems - Average Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars1, rag_means, rag_stds)):
        ax1.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontweight='bold')
    
    # Retrieval Systems Performance Overview
    ax2 = axes[0, 1]
    retrieval_metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']
    retrieval_means = [retrieval_df[metric].mean() for metric in retrieval_metrics]
    retrieval_stds = [retrieval_df[metric].std() for metric in retrieval_metrics]
    
    x_pos = np.arange(len(retrieval_metrics))
    bars2 = ax2.bar(x_pos, retrieval_means, yerr=retrieval_stds, capsize=5,
                    color='lightcoral', alpha=0.7, label='Mean ± Std')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k'], rotation=45)
    ax2.set_title('Retrieval Systems - Average Performance', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars2, retrieval_means, retrieval_stds)):
        ax2.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontweight='bold')
    
    # System Count Comparison
    ax3 = axes[1, 0]
    system_counts = [len(rag_df), len(retrieval_df)]
    dataset_names = ['RAG Systems', 'Retrieval Systems']
    colors = ['skyblue', 'lightcoral']
    
    bars3 = ax3.bar(dataset_names, system_counts, color=colors, alpha=0.7)
    ax3.set_title('Number of Systems Evaluated', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars3, system_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, count + 0.1, str(count), 
                ha='center', fontweight='bold', fontsize=14)
    
    # Evaluation Type Distribution (for retrieval)
    ax4 = axes[1, 1]
    if 'type' in retrieval_df.columns:
        type_counts = retrieval_df['type'].value_counts()
        colors_pie = ['lightgreen', 'orange', 'purple'][:len(type_counts)]
        
        wedges, texts, autotexts = ax4.pie(type_counts.values, labels=type_counts.index, 
                                          autopct='%1.1f%%', colors=colors_pie)
        ax4.set_title('Retrieval Evaluation Types', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax4.text(0.5, 0.5, 'No evaluation type\ndata available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Retrieval Evaluation Types', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_systems_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'combined_systems_overview.pdf', bbox_inches='tight')
    plt.show()

def create_cross_system_analysis(rag_df, retrieval_df, output_dir):
    """Analyze potential overlaps and relationships between RAG and retrieval systems."""
    
    # Try to find common systems or patterns
    rag_systems = set(rag_df['run_clean'].str.lower())
    retrieval_systems = set(retrieval_df['experiment_clean'].str.lower())
    
    # Look for systems that might be related (containing similar keywords)
    common_keywords = []
    for rag_sys in rag_systems:
        for ret_sys in retrieval_systems:
            # Check for common words (excluding common stopwords)
            rag_words = set(rag_sys.replace('-', ' ').replace('_', ' ').split())
            ret_words = set(ret_sys.replace('-', ' ').replace('_', ' ').split())
            
            common = rag_words.intersection(ret_words)
            common = common - {'the', 'and', 'or', 'with', 'of', 'a', 'an', 'in', 'on', 'at', 'to', 'for'}
            
            if len(common) >= 2:  # At least 2 common meaningful words
                common_keywords.append({
                    'rag_system': rag_sys,
                    'retrieval_system': ret_sys,
                    'common_words': list(common)
                })
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Performance Distribution Comparison
    ax1 = axes[0, 0]
    
    # Normalize RAG and retrieval metrics to [0,1] for comparison
    rag_norm = rag_df[['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']].mean(axis=1)
    retrieval_norm = retrieval_df[['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']].mean(axis=1)
    
    ax1.hist(rag_norm, bins=10, alpha=0.7, label='RAG Systems', color='skyblue', density=True)
    ax1.hist(retrieval_norm, bins=10, alpha=0.7, label='Retrieval Systems', color='lightcoral', density=True)
    ax1.set_xlabel('Average Normalized Performance', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Performance Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Technology Usage Analysis
    ax2 = axes[0, 1]
    
    # Extract technology keywords from system names
    tech_keywords = ['gemini', 'medgemma', 'cross-encoder', 'bi-encoder', 'rerank', 'medcpt', 'bge']
    rag_tech_counts = {}
    retrieval_tech_counts = {}
    
    for keyword in tech_keywords:
        rag_count = sum(1 for name in rag_df['run_clean'] if keyword.lower() in name.lower())
        retrieval_count = sum(1 for name in retrieval_df['experiment_clean'] if keyword.lower() in name.lower())
        if rag_count > 0 or retrieval_count > 0:
            rag_tech_counts[keyword] = rag_count
            retrieval_tech_counts[keyword] = retrieval_count
    
    technologies = list(rag_tech_counts.keys())
    rag_counts = [rag_tech_counts[tech] for tech in technologies]
    retrieval_counts = [retrieval_tech_counts[tech] for tech in technologies]
    
    x_pos = np.arange(len(technologies))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, rag_counts, width, label='RAG Systems', color='skyblue', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, retrieval_counts, width, label='Retrieval Systems', color='lightcoral', alpha=0.7)
    
    ax2.set_xlabel('Technology', fontsize=12)
    ax2.set_ylabel('Number of Systems', fontsize=12)
    ax2.set_title('Technology Usage Across System Types', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(technologies, rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Performance vs Complexity Analysis
    ax3 = axes[1, 0]
    
    # Use system name length as a proxy for complexity
    rag_complexity = [len(name) for name in rag_df['run_clean']]
    retrieval_complexity = [len(name) for name in retrieval_df['experiment_clean']]
    
    ax3.scatter(rag_complexity, rag_norm, alpha=0.7, label='RAG Systems', 
               color='skyblue', s=100, edgecolors='black')
    ax3.scatter(retrieval_complexity, retrieval_norm, alpha=0.7, label='Retrieval Systems', 
               color='lightcoral', s=100, edgecolors='black')
    
    ax3.set_xlabel('System Name Length (Complexity Proxy)', fontsize=12)
    ax3.set_ylabel('Average Performance Score', fontsize=12)
    ax3.set_title('Performance vs System Complexity', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Common Systems Analysis
    ax4 = axes[1, 1]
    
    if common_keywords:
        # Show potential system relationships
        ax4.text(0.1, 0.9, 'Potentially Related Systems:', fontsize=12, fontweight='bold', 
                transform=ax4.transAxes)
        
        y_pos = 0.8
        for i, common in enumerate(common_keywords[:5]):  # Show top 5
            text = f"• {common['rag_system'][:20]} ↔ {common['retrieval_system'][:20]}"
            ax4.text(0.1, y_pos - i*0.12, text, fontsize=10, transform=ax4.transAxes)
            
            # Show common words
            words_text = f"  Common: {', '.join(common['common_words'])}"
            ax4.text(0.15, y_pos - i*0.12 - 0.04, words_text, fontsize=8, 
                    style='italic', color='gray', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No obvious system\nrelationships found', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    ax4.set_title('Cross-System Relationships', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_system_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cross_system_analysis.pdf', bbox_inches='tight')
    plt.show()

def create_performance_benchmark(rag_df, retrieval_df, output_dir):
    """Create benchmark comparison charts."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top Performers in Each Category
    ax1 = axes[0, 0]
    
    # Get top 3 performers from each dataset
    rag_top3 = rag_df.nlargest(3, 'avg_f1')[['run_clean', 'avg_f1']]
    retrieval_top3 = retrieval_df.nlargest(3, 'MRR@k')[['experiment_clean', 'MRR@k']]
    
    # Plot top RAG systems
    y_pos_rag = np.arange(len(rag_top3))
    bars_rag = ax1.barh(y_pos_rag, rag_top3['avg_f1'], color='skyblue', alpha=0.7)
    ax1.set_yticks(y_pos_rag)
    ax1.set_yticklabels(rag_top3['run_clean'], fontsize=10)
    ax1.set_xlabel('F1 Score', fontsize=12)
    ax1.set_title('Top 3 RAG Systems (by F1)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars_rag, rag_top3['avg_f1'])):
        ax1.text(value + 0.005, i, f'{value:.3f}', va='center', fontweight='bold')
    
    # Top Retrieval systems
    ax2 = axes[0, 1]
    y_pos_ret = np.arange(len(retrieval_top3))
    bars_ret = ax2.barh(y_pos_ret, retrieval_top3['MRR@k'], color='lightcoral', alpha=0.7)
    ax2.set_yticks(y_pos_ret)
    ax2.set_yticklabels(retrieval_top3['experiment_clean'], fontsize=10)
    ax2.set_xlabel('MRR@k Score', fontsize=12)
    ax2.set_title('Top 3 Retrieval Systems (by MRR@k)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars_ret, retrieval_top3['MRR@k'])):
        ax2.text(value + 0.01, i, f'{value:.3f}', va='center', fontweight='bold')
    
    # Performance Consistency Analysis
    ax3 = axes[1, 0]
    
    # Calculate coefficient of variation for each system type
    rag_metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    retrieval_metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']
    
    rag_cv = []
    for _, row in rag_df.iterrows():
        values = [row[metric] for metric in rag_metrics]
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        rag_cv.append(cv)
    
    retrieval_cv = []
    for _, row in retrieval_df.iterrows():
        values = [row[metric] for metric in retrieval_metrics]
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        retrieval_cv.append(cv)
    
    ax3.boxplot([rag_cv, retrieval_cv], labels=['RAG Systems', 'Retrieval Systems'])
    ax3.set_ylabel('Coefficient of Variation', fontsize=12)
    ax3.set_title('Performance Consistency\n(Lower = More Consistent)', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Scatter plot: Best metric vs Average performance
    ax4 = axes[1, 1]
    
    # For RAG: best individual metric vs average
    rag_best = rag_df[rag_metrics].max(axis=1)
    rag_avg = rag_df[rag_metrics].mean(axis=1)
    
    # For Retrieval: best individual metric vs average
    retrieval_best = retrieval_df[retrieval_metrics].max(axis=1)
    retrieval_avg = retrieval_df[retrieval_metrics].mean(axis=1)
    
    ax4.scatter(rag_avg, rag_best, alpha=0.7, label='RAG Systems', 
               color='skyblue', s=100, edgecolors='black')
    ax4.scatter(retrieval_avg, retrieval_best, alpha=0.7, label='Retrieval Systems', 
               color='lightcoral', s=100, edgecolors='black')
    
    # Add diagonal line for reference
    min_val = min(min(rag_avg), min(retrieval_avg))
    max_val = max(max(rag_best), max(retrieval_best))
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    ax4.set_xlabel('Average Performance', fontsize=12)
    ax4.set_ylabel('Best Individual Metric', fontsize=12)
    ax4.set_title('Best vs Average Performance', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_benchmark.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'performance_benchmark.pdf', bbox_inches='tight')
    plt.show()

def generate_combined_summary_report(rag_df, retrieval_df, output_dir):
    """Generate a comprehensive summary report."""
    report_data = {
        'Dataset': ['RAG Evaluation', 'Retrieval Experiments'],
        'Number of Systems': [len(rag_df), len(retrieval_df)],
        'Primary Metrics': ['F1, Faithfulness, Correctness, Completeness', 
                           'MRR@k, nDCG@k, Recall@k, Precision@k'],
        'Best System (Primary Metric)': [
            rag_df.loc[rag_df['avg_f1'].idxmax(), 'run_clean'],
            retrieval_df.loc[retrieval_df['MRR@k'].idxmax(), 'experiment_clean']
        ],
        'Best Score': [
            f"{rag_df['avg_f1'].max():.3f}",
            f"{retrieval_df['MRR@k'].max():.3f}"
        ],
        'Average Score': [
            f"{rag_df['avg_f1'].mean():.3f}",
            f"{retrieval_df['MRR@k'].mean():.3f}"
        ]
    }
    
    summary_df = pd.DataFrame(report_data)
    
    # Save to CSV
    summary_df.to_csv(output_dir / 'combined_evaluation_summary.csv', index=False)
    
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
            if i % 2 == 0:
                table[(i+1, j)].set_facecolor('#E8F5E8')
    
    plt.title('Combined Evaluation Summary Report', 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'combined_summary_report.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'combined_summary_report.pdf', bbox_inches='tight')
    plt.show()
    
    return summary_df

def main():
    """Main function to generate combined analysis dashboard."""
    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = script_dir
    
    print("🔄 Loading both datasets...")
    rag_df, retrieval_df = load_both_datasets(script_dir)
    
    print(f"✅ Loaded {len(rag_df)} RAG systems and {len(retrieval_df)} retrieval systems")
    print(f"📊 Generating combined visualizations in: {output_dir}")
    
    # Generate all visualizations
    print("🏠 Creating systems overview...")
    create_system_overview(rag_df, retrieval_df, output_dir)
    
    print("🔍 Creating cross-system analysis...")
    create_cross_system_analysis(rag_df, retrieval_df, output_dir)
    
    print("🏆 Creating performance benchmark...")
    create_performance_benchmark(rag_df, retrieval_df, output_dir)
    
    print("📋 Generating combined summary report...")
    summary_df = generate_combined_summary_report(rag_df, retrieval_df, output_dir)
    
    print("\n" + "="*80)
    print("📊 COMBINED ANALYSIS SUMMARY")
    print("="*80)
    print(f"RAG Systems: {len(rag_df)} evaluated")
    print(f"Retrieval Systems: {len(retrieval_df)} evaluated")
    print(f"Total Systems: {len(rag_df) + len(retrieval_df)}")
    
    print("\nBest performers:")
    print(f"  RAG (F1): {rag_df.loc[rag_df['avg_f1'].idxmax(), 'run_clean']} ({rag_df['avg_f1'].max():.3f})")
    print(f"  Retrieval (MRR@k): {retrieval_df.loc[retrieval_df['MRR@k'].idxmax(), 'experiment_clean']} ({retrieval_df['MRR@k'].max():.3f})")
    
    print(f"\n✅ All combined visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()