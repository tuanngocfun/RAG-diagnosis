#!/usr/bin/env python3
"""
Ranking Analysis Script
Generates specialized visualizations focused on system performance rankings and orderings.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_unique_rag_name(row):
    """Create unique, meaningful names for RAG systems."""
    name = row['run']
    
    name_mapping = {
        'text_size12k, token1024 (missing)': 'Gemini25-12k-1024tok',
        'gem25-gemini-rpm-5 (missing)': 'Gemini25-Base',
        'gemimi_rerank_rag_test (all)': 'Gemini-Rerank-RAG',
        'RTX3090 cross-enc MedCPT (final, all)': 'CrossEnc-MedCPT-Final',
        'bi-encoder reranker (all)': 'BiEncoder-Rerank',
        'cross-encoder reranker (all)': 'CrossEnc-Rerank',
        'cross-encoder MedCPT pool6 top6 (all)': 'CrossEnc-MedCPT-Pool6'
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
        'Standalone Gemini2.5 (no retriever)': 'Gemini25-Standalone'
    }
    
    return name_mapping.get(name, name[:20])

def load_and_prepare_data(script_dir):
    """Load and prepare both datasets with ranking focus."""
    rag_path = script_dir / 'rag_gemini_eval_comparison_2025-09-29.csv'
    retrieval_path = script_dir / 'retrieval_experiments_comparison_updated.csv'
    
    # Load RAG data
    rag_df = pd.read_csv(rag_path)
    rag_df['system_name'] = rag_df.apply(create_unique_rag_name, axis=1)
    
    # Load Retrieval data  
    retrieval_df = pd.read_csv(retrieval_path)
    retrieval_df['system_name'] = retrieval_df.apply(create_unique_retrieval_name, axis=1)
    retrieval_df = retrieval_df[(retrieval_df[['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']] != 0).any(axis=1)]
    
    return rag_df, retrieval_df

def create_ranking_consistency_analysis(rag_df, output_dir):
    """Analyze ranking consistency across different RAG metrics."""
    metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    metric_names = ['F1', 'Faithfulness', 'Correctness', 'Completeness']
    
    # Calculate rankings for each metric
    rankings = {}
    for metric in metrics:
        rankings[metric] = rag_df[metric].rank(ascending=False, method='min')
    
    ranking_df = pd.DataFrame(rankings)
    ranking_df['system_name'] = rag_df['system_name'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Ranking correlation heatmap
    ax1 = axes[0, 0]
    corr_matrix = ranking_df[metrics].corr(method='spearman')
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                xticklabels=metric_names, yticklabels=metric_names, ax=ax1)
    ax1.set_title('Ranking Correlation Matrix\n(Spearman Correlation)', fontsize=14, fontweight='bold')
    
    # Ranking consistency plot
    ax2 = axes[0, 1]
    
    # Calculate ranking variance for each system
    ranking_variance = ranking_df[metrics].var(axis=1)
    systems_sorted = ranking_df.loc[ranking_variance.sort_values().index]
    
    y_pos = np.arange(len(systems_sorted))
    bars = ax2.barh(y_pos, ranking_variance.sort_values(), color='lightblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(systems_sorted['system_name'], fontsize=10)
    ax2.set_xlabel('Ranking Variance', fontsize=12)
    ax2.set_title('Ranking Consistency\n(Lower Variance = More Consistent)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add variance values
    for i, (bar, var) in enumerate(zip(bars, ranking_variance.sort_values())):
        ax2.text(var + 0.1, i, f'{var:.1f}', va='center', fontsize=9, fontweight='bold')
    
    # Parallel coordinates plot for rankings
    ax3 = axes[1, 0]
    
    # Normalize rankings to 0-1 for better visualization
    ranking_norm = ranking_df[metrics].copy()
    for metric in metrics:
        ranking_norm[metric] = (ranking_norm[metric] - 1) / (len(rag_df) - 1)
    
    # Plot lines for each system
    colors = sns.color_palette("husl", len(ranking_norm))
    for i, (_, row) in enumerate(ranking_norm.iterrows()):
        ax3.plot(range(len(metrics)), [row[metric] for metric in metrics], 
                'o-', alpha=0.7, color=colors[i], linewidth=2)
    
    ax3.set_xticks(range(len(metrics)))
    ax3.set_xticklabels(metric_names, rotation=45)
    ax3.set_ylabel('Normalized Ranking (0=Best, 1=Worst)', fontsize=12)
    ax3.set_title('Ranking Patterns Across Metrics', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # Invert so best (0) is at top
    
    # Winner analysis
    ax4 = axes[1, 1]
    
    # Count wins (rank 1) for each system
    wins = {}
    for metric in metrics:
        winner_idx = rag_df[metric].idxmax()
        winner_name = rag_df.loc[winner_idx, 'system_name']
        wins[winner_name] = wins.get(winner_name, 0) + 1
    
    if wins:
        systems = list(wins.keys())
        win_counts = list(wins.values())
        
        bars = ax4.bar(systems, win_counts, color='gold', alpha=0.8)
        ax4.set_ylabel('Number of Metric Wins', fontsize=12)
        ax4.set_title('Systems with #1 Rankings', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add win count labels
        for bar, count in zip(bars, win_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, count + 0.05, str(count), 
                    ha='center', fontweight='bold', fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No clear winners found', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rag_ranking_consistency.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'rag_ranking_consistency.pdf', bbox_inches='tight')
    plt.show()

def create_retrieval_ranking_analysis(retrieval_df, output_dir):
    """Analyze ranking patterns in retrieval systems."""
    metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']
    
    # Calculate rankings
    rankings = {}
    for metric in metrics:
        rankings[metric] = retrieval_df[metric].rank(ascending=False, method='min')
    
    ranking_df = pd.DataFrame(rankings)
    ranking_df['system_name'] = retrieval_df['system_name'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Ranking vs Success Rate
    ax1 = axes[0, 0]
    
    # Calculate average ranking and success rate
    avg_ranking = ranking_df[metrics].mean(axis=1)
    success_rate = retrieval_df['successful_queries'] / retrieval_df['total_queries']
    
    scatter = ax1.scatter(success_rate, avg_ranking, s=100, alpha=0.7, 
                         c=range(len(retrieval_df)), cmap='viridis', edgecolors='black')
    
    # Add system labels
    for i, (sr, ar, name) in enumerate(zip(success_rate, avg_ranking, retrieval_df['system_name'])):
        ax1.annotate(name[:15], (sr, ar), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    ax1.set_xlabel('Success Rate', fontsize=12)
    ax1.set_ylabel('Average Ranking (Lower = Better)', fontsize=12)
    ax1.set_title('Ranking vs Success Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Evaluation type ranking comparison
    ax2 = axes[0, 1]
    
    if 'type' in retrieval_df.columns:
        eval_types = retrieval_df['type'].unique()
        for eval_type in eval_types:
            type_data = retrieval_df[retrieval_df['type'] == eval_type]
            type_rankings = []
            for metric in metrics:
                if type_data[metric].max() > 0:
                    type_rankings.extend(type_data[metric].rank(ascending=False))
            
            if type_rankings:
                ax2.hist(type_rankings, alpha=0.6, label=eval_type, bins=10, density=True)
        
        ax2.set_xlabel('Ranking', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Ranking Distribution by Evaluation Type', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No evaluation type data', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
    
    # Performance vs Query Load
    ax3 = axes[1, 0]
    
    # Use total queries as proxy for system load
    total_queries = retrieval_df['total_queries']
    performance_score = retrieval_df[metrics].mean(axis=1)
    
    scatter2 = ax3.scatter(total_queries, performance_score, s=100, alpha=0.7,
                          c=avg_ranking, cmap='RdYlGn', edgecolors='black')
    
    # Add colorbar
    cbar = plt.colorbar(scatter2, ax=ax3)
    cbar.set_label('Average Ranking', fontsize=10)
    
    ax3.set_xlabel('Total Queries Processed', fontsize=12)
    ax3.set_ylabel('Average Performance Score', fontsize=12)
    ax3.set_title('Performance vs Query Load\n(Color = Avg Ranking)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Top performer stability
    ax4 = axes[1, 1]
    
    # Calculate how often each system appears in top 3 for each metric
    top3_appearances = {}
    for metric in metrics:
        top3_indices = retrieval_df.nlargest(3, metric).index
        for idx in top3_indices:
            system_name = retrieval_df.loc[idx, 'system_name']
            top3_appearances[system_name] = top3_appearances.get(system_name, 0) + 1
    
    if top3_appearances:
        systems = list(top3_appearances.keys())
        appearances = list(top3_appearances.values())
        
        bars = ax4.bar(systems, appearances, color='lightgreen', alpha=0.8)
        ax4.set_ylabel('Top-3 Appearances', fontsize=12)
        ax4.set_title('Top Performer Stability', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add appearance count labels
        for bar, count in zip(bars, appearances):
            ax4.text(bar.get_x() + bar.get_width()/2, count + 0.05, str(count),
                    ha='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_ranking_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'retrieval_ranking_analysis.pdf', bbox_inches='tight')
    plt.show()

def create_cross_dataset_ranking_comparison(rag_df, retrieval_df, output_dir):
    """Compare ranking patterns across both datasets."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Ranking distribution comparison
    ax1 = axes[0, 0]
    
    # Calculate ranking distributions
    rag_metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    retrieval_metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']
    
    rag_rankings = []
    for metric in rag_metrics:
        rag_rankings.extend(rag_df[metric].rank(ascending=False, method='min'))
    
    retrieval_rankings = []
    for metric in retrieval_metrics:
        retrieval_rankings.extend(retrieval_df[metric].rank(ascending=False, method='min'))
    
    ax1.hist(rag_rankings, bins=15, alpha=0.7, label='RAG Systems', color='skyblue', density=True)
    ax1.hist(retrieval_rankings, bins=15, alpha=0.7, label='Retrieval Systems', color='lightcoral', density=True)
    ax1.set_xlabel('Ranking Position', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Ranking Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance spread analysis
    ax2 = axes[0, 1]
    
    # Calculate coefficient of variation for rankings
    rag_ranking_cv = []
    for _, row in rag_df.iterrows():
        rankings = [row[metric] for metric in rag_metrics]
        rankings_normalized = stats.rankdata(rankings, method='min')
        cv = np.std(rankings_normalized) / np.mean(rankings_normalized)
        rag_ranking_cv.append(cv)
    
    retrieval_ranking_cv = []
    for _, row in retrieval_df.iterrows():
        rankings = [row[metric] for metric in retrieval_metrics]
        rankings_normalized = stats.rankdata(rankings, method='min')
        cv = np.std(rankings_normalized) / np.mean(rankings_normalized)
        retrieval_ranking_cv.append(cv)
    
    ax2.boxplot([rag_ranking_cv, retrieval_ranking_cv], 
                labels=['RAG Systems', 'Retrieval Systems'])
    ax2.set_ylabel('Ranking Coefficient of Variation', fontsize=12)
    ax2.set_title('Ranking Consistency Comparison\n(Lower = More Consistent)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # System complexity vs ranking stability
    ax3 = axes[1, 0]
    
    # Use system name length as complexity proxy
    rag_complexity = [len(name) for name in rag_df['system_name']]
    retrieval_complexity = [len(name) for name in retrieval_df['system_name']]
    
    ax3.scatter(rag_complexity, rag_ranking_cv, alpha=0.7, label='RAG Systems', 
               color='skyblue', s=80, edgecolors='black')
    ax3.scatter(retrieval_complexity, retrieval_ranking_cv, alpha=0.7, label='Retrieval Systems', 
               color='lightcoral', s=80, edgecolors='black')
    
    ax3.set_xlabel('System Name Length (Complexity)', fontsize=12)
    ax3.set_ylabel('Ranking CV', fontsize=12)
    ax3.set_title('Complexity vs Ranking Stability', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Overall ranking quality assessment
    ax4 = axes[1, 1]
    
    # Create a "ranking quality" metric based on top performers
    rag_top_scores = []
    for metric in rag_metrics:
        rag_top_scores.append(rag_df[metric].max())
    
    retrieval_top_scores = []
    for metric in retrieval_metrics:
        retrieval_top_scores.append(retrieval_df[metric].max())
    
    datasets = ['RAG Evaluation', 'Retrieval Experiments']
    avg_top_scores = [np.mean(rag_top_scores), np.mean(retrieval_top_scores)]
    std_top_scores = [np.std(rag_top_scores), np.std(retrieval_top_scores)]
    
    bars = ax4.bar(datasets, avg_top_scores, yerr=std_top_scores, capsize=5,
                   color=['skyblue', 'lightcoral'], alpha=0.7)
    ax4.set_ylabel('Average Top Performance Score', fontsize=12)
    ax4.set_title('Peak Performance Comparison', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, score, std in zip(bars, avg_top_scores, std_top_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, score + std + 0.02, 
                f'{score:.3f}±{std:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_dataset_ranking.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cross_dataset_ranking.pdf', bbox_inches='tight')
    plt.show()

def create_ranking_evolution_timeline(rag_df, output_dir):
    """Create timeline showing ranking evolution for RAG systems."""
    if 'timestamp' not in rag_df.columns:
        print("⚠️ No timestamp data available for timeline analysis")
        return
    
    # Convert timestamp and sort
    rag_df_sorted = rag_df.copy()
    rag_df_sorted['timestamp'] = pd.to_datetime(rag_df_sorted['timestamp'])
    rag_df_sorted = rag_df_sorted.sort_values('timestamp')
    
    metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    colors = sns.color_palette("husl", len(rag_df_sorted))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Calculate cumulative best score over time
        cumulative_best = []
        timestamps = []
        
        best_so_far = 0
        for _, row in rag_df_sorted.iterrows():
            current_score = row[metric]
            if current_score > best_so_far:
                best_so_far = current_score
            cumulative_best.append(best_so_far)
            timestamps.append(row['timestamp'])
        
        # Plot cumulative best
        ax.plot(timestamps, cumulative_best, 'b-', linewidth=3, label='Best Score So Far')
        
        # Plot individual system scores
        ax.scatter(rag_df_sorted['timestamp'], rag_df_sorted[metric], 
                  alpha=0.7, s=80, c=colors, edgecolors='black', label='Individual Systems')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{metric.replace("avg_", "").title()} Score', fontsize=12)
        ax.set_title(f'{metric.replace("avg_", "").title()} Evolution Over Time', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ranking_evolution_timeline.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ranking_evolution_timeline.pdf', bbox_inches='tight')
    plt.show()

def generate_ranking_summary_statistics(rag_df, retrieval_df, output_dir):
    """Generate comprehensive ranking statistics."""
    
    # RAG ranking stats
    rag_metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    rag_stats = {}
    
    for metric in rag_metrics:
        rankings = rag_df[metric].rank(ascending=False, method='min')
        rag_stats[metric] = {
            'winner': rag_df.loc[rag_df[metric].idxmax(), 'system_name'],
            'winner_score': rag_df[metric].max(),
            'median_rank': rankings.median(),
            'rank_range': rankings.max() - rankings.min(),
            'kendall_tau': None  # Will calculate with other metrics
        }
    
    # Retrieval ranking stats
    retrieval_metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']
    retrieval_stats = {}
    
    for metric in retrieval_metrics:
        rankings = retrieval_df[metric].rank(ascending=False, method='min')
        retrieval_stats[metric] = {
            'winner': retrieval_df.loc[retrieval_df[metric].idxmax(), 'system_name'],
            'winner_score': retrieval_df[metric].max(),
            'median_rank': rankings.median(),
            'rank_range': rankings.max() - rankings.min()
        }
    
    # Create summary tables
    rag_summary = pd.DataFrame(rag_stats).T
    retrieval_summary = pd.DataFrame(retrieval_stats).T
    
    # Save to CSV
    rag_summary.to_csv(output_dir / 'rag_ranking_summary.csv')
    retrieval_summary.to_csv(output_dir / 'retrieval_ranking_summary.csv')
    
    print(f"📊 RAG Ranking Summary:")
    print(rag_summary.round(3))
    print(f"\n📊 Retrieval Ranking Summary:")
    print(retrieval_summary.round(3))
    
    return rag_summary, retrieval_summary

def main():
    """Main function to generate all ranking analysis visualizations."""
    script_dir = Path(__file__).parent
    output_dir = script_dir
    
    print("🔄 Loading datasets for ranking analysis...")
    rag_df, retrieval_df = load_and_prepare_data(script_dir)
    
    print(f"✅ Loaded {len(rag_df)} RAG systems and {len(retrieval_df)} retrieval systems")
    print(f"📊 Generating ranking visualizations in: {output_dir}")
    
    # Generate all ranking analyses
    print("📈 Creating RAG ranking consistency analysis...")
    create_ranking_consistency_analysis(rag_df, output_dir)
    
    print("🔍 Creating retrieval ranking analysis...")
    create_retrieval_ranking_analysis(retrieval_df, output_dir)
    
    print("⚖️ Creating cross-dataset ranking comparison...")
    create_cross_dataset_ranking_comparison(rag_df, retrieval_df, output_dir)
    
    print("⏰ Creating ranking evolution timeline...")
    create_ranking_evolution_timeline(rag_df, output_dir)
    
    print("📋 Generating ranking summary statistics...")
    rag_summary, retrieval_summary = generate_ranking_summary_statistics(rag_df, retrieval_df, output_dir)
    
    print("\n" + "="*70)
    print("🏆 RANKING ANALYSIS SUMMARY")
    print("="*70)
    
    # Find most consistent performers
    rag_metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    rag_rankings = pd.DataFrame()
    for metric in rag_metrics:
        rag_rankings[metric] = rag_df[metric].rank(ascending=False, method='min')
    rag_rankings['system_name'] = rag_df['system_name']
    rag_rankings['rank_variance'] = rag_rankings[rag_metrics].var(axis=1)
    
    most_consistent = rag_rankings.loc[rag_rankings['rank_variance'].idxmin(), 'system_name']
    print(f"Most consistent RAG system: {most_consistent}")
    
    # Find top overall performers
    rag_avg_rank = rag_rankings[rag_metrics].mean(axis=1)
    best_overall = rag_rankings.loc[rag_avg_rank.idxmin(), 'system_name']
    print(f"Best overall RAG system: {best_overall}")
    
    retrieval_metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']
    retrieval_avg_perf = retrieval_df[retrieval_metrics].mean(axis=1)
    best_retrieval = retrieval_df.loc[retrieval_avg_perf.idxmax(), 'system_name']
    print(f"Best overall retrieval system: {best_retrieval}")
    
    print(f"\n✅ All ranking visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()