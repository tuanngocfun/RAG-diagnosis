#!/usr/bin/env python3
"""
Best Experiments Comparison Table Generator
Creates publication-ready tables and visualizations highlighting which experiments
perform best in which metrics for both retrieval and RAG evaluations.
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
        'text_size12k, token1024 (missing)': 'Gemini25-12k-1024tok',
        'gem25-gemini-rpm-5 (missing)': 'Gemini25-Base',
        'gemimi_rerank_rag_test (all)': 'Gemini-Rerank-RAG',
        'RTX3090 cross-enc MedCPT (final, all)': 'CrossEnc-MedCPT-Final',
        'bi-encoder reranker (all)': 'BiEncoder-Rerank',
        'cross-encoder reranker (all)': 'CrossEnc-Rerank',
        'cross-encoder MedCPT pool6 top6 (all)': 'CrossEnc-MedCPT-Pool6',
        'MedGemma4B standalone (all)': 'MedGemma4B-Standalone'
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

def create_retrieval_best_systems_table(retrieval_df, output_dir):
    """Create table showing best retrieval systems for each metric."""
    # Remove zero-performance systems
    retrieval_df = retrieval_df[(retrieval_df[['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']] != 0).any(axis=1)].copy()
    retrieval_df['system_name'] = retrieval_df.apply(create_unique_retrieval_name, axis=1)
    
    metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k', 'doc_recall@k', 'text_coverage@k']
    metric_display_names = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k', 'Doc Recall@k', 'Text Coverage@k']
    
    # Find top 3 systems for each metric
    results = []
    for metric, display_name in zip(metrics, metric_display_names):
        if metric in retrieval_df.columns and retrieval_df[metric].max() > 0:
            top3 = retrieval_df.nlargest(3, metric)
            
            for rank, (_, row) in enumerate(top3.iterrows(), 1):
                results.append({
                    'Metric': display_name,
                    'Rank': rank,
                    'System': row['system_name'],
                    'Score': f"{row[metric]:.4f}",
                    'Type': row['type'] if 'type' in row else 'N/A'
                })
    
    results_df = pd.DataFrame(results)
    
    # Create pivot table for better visualization
    pivot_table = results_df.pivot_table(
        index='Metric', 
        columns='Rank', 
        values='System', 
        aggfunc='first'
    )
    
    # Add scores
    scores_data = []
    for metric, display_name in zip(metrics, metric_display_names):
        if metric in retrieval_df.columns and retrieval_df[metric].max() > 0:
            top3 = retrieval_df.nlargest(3, metric)
            scores = [f"{row[metric]:.4f}" for _, row in top3.iterrows()]
            scores_data.append({
                'Metric': display_name,
                '1st': top3.iloc[0]['system_name'] + f"\n({scores[0]})",
                '2nd': top3.iloc[1]['system_name'] + f"\n({scores[1]})" if len(scores) > 1 else '',
                '3rd': top3.iloc[2]['system_name'] + f"\n({scores[2]})" if len(scores) > 2 else ''
            })
    
    final_table_df = pd.DataFrame(scores_data)
    
    # Save to CSV
    final_table_df.to_csv(output_dir / 'retrieval_best_systems_comparison.csv', index=False)
    results_df.to_csv(output_dir / 'retrieval_best_systems_detailed.csv', index=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=final_table_df.values,
                    colLabels=final_table_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.26, 0.26, 0.26])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)
    
    # Style the table - header
    for i in range(len(final_table_df.columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Style the table - cells with ranking colors
    for i in range(len(final_table_df)):
        # Metric name column
        table[(i+1, 0)].set_facecolor('#E8F4F8')
        table[(i+1, 0)].set_text_props(weight='bold')
        
        # 1st place - gold
        table[(i+1, 1)].set_facecolor('#FFD700')
        table[(i+1, 1)].set_text_props(weight='bold', fontsize=9)
        
        # 2nd place - silver
        if len(final_table_df.columns) > 2:
            table[(i+1, 2)].set_facecolor('#C0C0C0')
            table[(i+1, 2)].set_text_props(fontsize=9)
        
        # 3rd place - bronze
        if len(final_table_df.columns) > 3:
            table[(i+1, 3)].set_facecolor('#CD7F32')
            table[(i+1, 3)].set_text_props(fontsize=9)
    
    plt.title('🏆 Best Retrieval Systems by Metric (Top 3)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.figtext(0.5, 0.02, 'Gold = 1st, Silver = 2nd, Bronze = 3rd', 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_best_systems_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'retrieval_best_systems_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    return final_table_df

def create_rag_best_systems_table(rag_df, output_dir):
    """Create table showing best RAG systems for each metric."""
    rag_df['system_name'] = rag_df.apply(create_unique_rag_name, axis=1)
    
    metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    metric_display_names = ['F1 Score', 'Faithfulness', 'Correctness', 'Completeness']
    
    # Find top 3 systems for each metric
    results = []
    for metric, display_name in zip(metrics, metric_display_names):
        top3 = rag_df.nlargest(3, metric)
        
        for rank, (_, row) in enumerate(top3.iterrows(), 1):
            results.append({
                'Metric': display_name,
                'Rank': rank,
                'System': row['system_name'],
                'Score': f"{row[metric]:.4f}"
            })
    
    results_df = pd.DataFrame(results)
    
    # Create final table with scores
    scores_data = []
    for metric, display_name in zip(metrics, metric_display_names):
        top3 = rag_df.nlargest(3, metric)
        scores = [f"{row[metric]:.4f}" for _, row in top3.iterrows()]
        scores_data.append({
            'Metric': display_name,
            '1st': top3.iloc[0]['system_name'] + f"\n({scores[0]})",
            '2nd': top3.iloc[1]['system_name'] + f"\n({scores[1]})" if len(scores) > 1 else '',
            '3rd': top3.iloc[2]['system_name'] + f"\n({scores[2]})" if len(scores) > 2 else ''
        })
    
    final_table_df = pd.DataFrame(scores_data)
    
    # Save to CSV
    final_table_df.to_csv(output_dir / 'rag_best_systems_comparison.csv', index=False)
    results_df.to_csv(output_dir / 'rag_best_systems_detailed.csv', index=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=final_table_df.values,
                    colLabels=final_table_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.26, 0.26, 0.26])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)
    
    # Style the table - header
    for i in range(len(final_table_df.columns)):
        table[(0, i)].set_facecolor('#A23B72')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Style the table - cells with ranking colors
    for i in range(len(final_table_df)):
        # Metric name column
        table[(i+1, 0)].set_facecolor('#F8E8F0')
        table[(i+1, 0)].set_text_props(weight='bold')
        
        # 1st place - gold
        table[(i+1, 1)].set_facecolor('#FFD700')
        table[(i+1, 1)].set_text_props(weight='bold', fontsize=9)
        
        # 2nd place - silver
        if len(final_table_df.columns) > 2:
            table[(i+1, 2)].set_facecolor('#C0C0C0')
            table[(i+1, 2)].set_text_props(fontsize=9)
        
        # 3rd place - bronze
        if len(final_table_df.columns) > 3:
            table[(i+1, 3)].set_facecolor('#CD7F32')
            table[(i+1, 3)].set_text_props(fontsize=9)
    
    plt.title('🏆 Best RAG Systems by Metric (Top 3)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.figtext(0.5, 0.02, 'Gold = 1st, Silver = 2nd, Bronze = 3rd', 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rag_best_systems_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'rag_best_systems_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    return final_table_df

def create_overall_champion_table(retrieval_df, rag_df, output_dir):
    """Create overall champion table showing best system per metric across both evaluations."""
    # Process retrieval data
    retrieval_df = retrieval_df[(retrieval_df[['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']] != 0).any(axis=1)].copy()
    retrieval_df['system_name'] = retrieval_df.apply(create_unique_retrieval_name, axis=1)
    
    # Process RAG data
    rag_df['system_name'] = rag_df.apply(create_unique_rag_name, axis=1)
    
    # Retrieval champions
    retrieval_metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k', 'doc_recall@k', 'text_coverage@k']
    retrieval_champions = []
    
    for metric in retrieval_metrics:
        if metric in retrieval_df.columns and retrieval_df[metric].max() > 0:
            best = retrieval_df.loc[retrieval_df[metric].idxmax()]
            retrieval_champions.append({
                'Category': 'Retrieval',
                'Metric': metric,
                'Champion System': best['system_name'],
                'Score': f"{best[metric]:.4f}",
                'Type': best['type'] if 'type' in best else 'N/A'
            })
    
    # RAG champions
    rag_metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    rag_metric_names = ['F1 Score', 'Faithfulness', 'Correctness', 'Completeness']
    rag_champions = []
    
    for metric, display_name in zip(rag_metrics, rag_metric_names):
        best = rag_df.loc[rag_df[metric].idxmax()]
        rag_champions.append({
            'Category': 'RAG',
            'Metric': display_name,
            'Champion System': best['system_name'],
            'Score': f"{best[metric]:.4f}",
            'Type': 'Generation'
        })
    
    # Combine
    all_champions = retrieval_champions + rag_champions
    champions_df = pd.DataFrame(all_champions)
    
    # Save to CSV
    champions_df.to_csv(output_dir / 'overall_champions_table.csv', index=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=champions_df.values,
                    colLabels=champions_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.2, 0.35, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style the table - header
    for i in range(len(champions_df.columns)):
        table[(0, i)].set_facecolor('#1A5490')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style the table - cells
    for i in range(len(champions_df)):
        category = champions_df.iloc[i]['Category']
        
        if category == 'Retrieval':
            color = '#E8F4F8'
            text_color = '#2E86AB'
        else:  # RAG
            color = '#F8E8F0'
            text_color = '#A23B72'
        
        # Category column
        table[(i+1, 0)].set_facecolor(color)
        table[(i+1, 0)].set_text_props(weight='bold', color=text_color)
        
        # Metric column
        table[(i+1, 1)].set_facecolor(color)
        
        # Champion system column - highlight
        table[(i+1, 2)].set_facecolor('#FFD700')
        table[(i+1, 2)].set_text_props(weight='bold', fontsize=11)
        
        # Score column
        table[(i+1, 3)].set_facecolor(color)
        table[(i+1, 3)].set_text_props(weight='bold')
        
        # Type column
        table[(i+1, 4)].set_facecolor(color)
    
    plt.title('🏆 Overall Champions: Best System for Each Metric', 
              fontsize=20, fontweight='bold', pad=20)
    plt.figtext(0.5, 0.02, 'Champion systems highlighted in gold', 
                ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_champions_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'overall_champions_table.pdf', bbox_inches='tight')
    plt.close()
    
    return champions_df

def create_metric_winners_heatmap(retrieval_df, rag_df, output_dir):
    """Create heatmap showing which system wins in which metric."""
    # Process retrieval data
    retrieval_df = retrieval_df[(retrieval_df[['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']] != 0).any(axis=1)].copy()
    retrieval_df['system_name'] = retrieval_df.apply(create_unique_retrieval_name, axis=1)
    
    # Process RAG data
    rag_df['system_name'] = rag_df.apply(create_unique_rag_name, axis=1)
    
    # Create combined dataset for heatmap
    all_systems = list(set(list(retrieval_df['system_name']) + list(rag_df['system_name'])))
    
    # Metrics to track
    all_metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k', 
                   'F1', 'Faithfulness', 'Correctness', 'Completeness']
    
    # Create wins matrix
    wins_matrix = pd.DataFrame(0, index=all_systems, columns=all_metrics)
    
    # Fill retrieval wins
    for metric in ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']:
        if metric in retrieval_df.columns and retrieval_df[metric].max() > 0:
            winner = retrieval_df.loc[retrieval_df[metric].idxmax(), 'system_name']
            wins_matrix.loc[winner, metric] = 1
    
    # Fill RAG wins
    rag_metric_map = {
        'avg_f1': 'F1',
        'avg_faithfulness': 'Faithfulness',
        'avg_correctness': 'Correctness',
        'avg_completeness': 'Completeness'
    }
    
    for orig_metric, display_metric in rag_metric_map.items():
        winner = rag_df.loc[rag_df[orig_metric].idxmax(), 'system_name']
        wins_matrix.loc[winner, display_metric] = 1
    
    # Remove systems with no wins
    wins_matrix = wins_matrix.loc[wins_matrix.sum(axis=1) > 0]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(wins_matrix, 
                annot=True, 
                fmt='d',
                cmap=['#FFFFFF', '#FFD700'],
                cbar=False,
                linewidths=2,
                linecolor='gray',
                square=False,
                ax=ax)
    
    ax.set_title('🎯 Metric Winners Heatmap\n(1 = Champion, 0 = Not Champion)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Systems', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_winners_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'metric_winners_heatmap.pdf', bbox_inches='tight')
    plt.close()
    
    # Save wins summary
    wins_summary = wins_matrix.sum(axis=1).sort_values(ascending=False)
    wins_summary_df = pd.DataFrame({
        'System': wins_summary.index,
        'Total Wins': wins_summary.values
    })
    wins_summary_df.to_csv(output_dir / 'metric_wins_summary.csv', index=False)
    
    return wins_matrix

def create_comparative_summary(retrieval_df, rag_df, output_dir):
    """Create comprehensive comparative summary table."""
    # Process data
    retrieval_df = retrieval_df[(retrieval_df[['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']] != 0).any(axis=1)].copy()
    retrieval_df['system_name'] = retrieval_df.apply(create_unique_retrieval_name, axis=1)
    rag_df['system_name'] = rag_df.apply(create_unique_rag_name, axis=1)
    
    summary_data = []
    
    # Retrieval summary
    retrieval_metrics = ['MRR@k', 'nDCG@k', 'Recall@k', 'Precision@k']
    for metric in retrieval_metrics:
        if metric in retrieval_df.columns and retrieval_df[metric].max() > 0:
            best = retrieval_df.loc[retrieval_df[metric].idxmax()]
            avg = retrieval_df[metric].mean()
            std = retrieval_df[metric].std()
            
            summary_data.append({
                'Evaluation': 'Retrieval',
                'Metric': metric,
                'Best System': best['system_name'],
                'Best Score': f"{best[metric]:.4f}",
                'Average': f"{avg:.4f}",
                'Std Dev': f"{std:.4f}",
                'Systems Evaluated': len(retrieval_df)
            })
    
    # RAG summary
    rag_metrics = ['avg_f1', 'avg_faithfulness', 'avg_correctness', 'avg_completeness']
    rag_display = ['F1 Score', 'Faithfulness', 'Correctness', 'Completeness']
    
    for metric, display in zip(rag_metrics, rag_display):
        best = rag_df.loc[rag_df[metric].idxmax()]
        avg = rag_df[metric].mean()
        std = rag_df[metric].std()
        
        summary_data.append({
            'Evaluation': 'RAG',
            'Metric': display,
            'Best System': best['system_name'],
            'Best Score': f"{best[metric]:.4f}",
            'Average': f"{avg:.4f}",
            'Std Dev': f"{std:.4f}",
            'Systems Evaluated': len(rag_df)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_df.to_csv(output_dir / 'comparative_summary_table.csv', index=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12, 0.15, 0.25, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style the table - header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#1A1A1A')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Style the table - cells
    for i in range(len(summary_df)):
        evaluation = summary_df.iloc[i]['Evaluation']
        
        if evaluation == 'Retrieval':
            bg_color = '#E8F4F8'
            highlight_color = '#2E86AB'
        else:
            bg_color = '#F8E8F0'
            highlight_color = '#A23B72'
        
        # Evaluation type
        table[(i+1, 0)].set_facecolor(bg_color)
        table[(i+1, 0)].set_text_props(weight='bold', color=highlight_color)
        
        # Metric
        table[(i+1, 1)].set_facecolor(bg_color)
        table[(i+1, 1)].set_text_props(weight='bold')
        
        # Best System - highlight
        table[(i+1, 2)].set_facecolor('#FFD700')
        table[(i+1, 2)].set_text_props(weight='bold', fontsize=10)
        
        # Other columns
        for j in range(3, 7):
            table[(i+1, j)].set_facecolor(bg_color)
            if j == 3:  # Best Score
                table[(i+1, j)].set_text_props(weight='bold')
    
    plt.title('📊 Comparative Summary: Retrieval vs RAG Experiments', 
              fontsize=20, fontweight='bold', pad=20)
    plt.figtext(0.5, 0.02, 'Publication-ready comparison of all experiments', 
                ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparative_summary_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparative_summary_table.pdf', bbox_inches='tight')
    plt.close()
    
    return summary_df

def main():
    """Main function to generate all comparison tables and visualizations."""
    # Setup paths
    script_dir = Path(__file__).parent / 'updated' / 'best_exp'
    retrieval_path = script_dir / 'Final_Retrieval_Experiments_Comparison.csv'
    rag_path = script_dir / 'Updated_RAG_experiment_comparison_with_MedGemma4B_standalone.csv'
    output_dir = script_dir
    
    print("🔄 Loading experiment data...")
    retrieval_df = pd.read_csv(retrieval_path)
    rag_df = pd.read_csv(rag_path)
    
    print(f"✅ Loaded {len(retrieval_df)} retrieval experiments")
    print(f"✅ Loaded {len(rag_df)} RAG experiments")
    print(f"📊 Generating comparison tables in: {output_dir}")
    
    # Generate all tables
    print("\n🏆 Creating retrieval best systems table...")
    retrieval_best = create_retrieval_best_systems_table(retrieval_df, output_dir)
    print("   ✅ Saved: retrieval_best_systems_comparison.png/pdf/csv")
    
    print("\n🏆 Creating RAG best systems table...")
    rag_best = create_rag_best_systems_table(rag_df, output_dir)
    print("   ✅ Saved: rag_best_systems_comparison.png/pdf/csv")
    
    print("\n👑 Creating overall champions table...")
    champions = create_overall_champion_table(retrieval_df, rag_df, output_dir)
    print("   ✅ Saved: overall_champions_table.png/pdf/csv")
    
    print("\n🎯 Creating metric winners heatmap...")
    winners_heatmap = create_metric_winners_heatmap(retrieval_df, rag_df, output_dir)
    print("   ✅ Saved: metric_winners_heatmap.png/pdf")
    
    print("\n📊 Creating comparative summary table...")
    comparative = create_comparative_summary(retrieval_df, rag_df, output_dir)
    print("   ✅ Saved: comparative_summary_table.png/pdf/csv")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 EXPERIMENTS COMPARISON SUMMARY")
    print("="*80)
    
    print("\n🏆 Top Retrieval Systems:")
    print(retrieval_best.to_string(index=False))
    
    print("\n🏆 Top RAG Systems:")
    print(rag_best.to_string(index=False))
    
    print("\n👑 Overall Champions (by metric):")
    print(champions[['Metric', 'Champion System', 'Score']].to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ ALL COMPARISON TABLES GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\n📁 Output files saved to:", output_dir)
    print("\n📋 Generated files:")
    print("   • retrieval_best_systems_comparison.png/pdf/csv")
    print("   • rag_best_systems_comparison.png/pdf/csv")
    print("   • overall_champions_table.png/pdf/csv")
    print("   • metric_winners_heatmap.png/pdf")
    print("   • comparative_summary_table.png/pdf/csv")
    print("   • metric_wins_summary.csv")
    print("\n📝 All tables are publication-ready for your paper!")

if __name__ == "__main__":
    main()
