import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import subprocess
import os
from datetime import datetime

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure matplotlib to handle font warnings
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def get_files_from_directory(directory_path):
    """
    Get file list from directory using shell command
    """
    try:
        # Run ls command and capture output
        result = subprocess.run(['ls', directory_path], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        # Split output into list of files
        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        print(f"Found {len(files)} files in {directory_path}")
        return files
        
    except subprocess.CalledProcessError as e:
        print(f"Error accessing directory {directory_path}: {e}")
        return []
    except FileNotFoundError:
        print("'ls' command not found. Using Python's os.listdir instead...")
        try:
            files = os.listdir(directory_path)
            print(f"Found {len(files)} files in {directory_path}")
            return files
        except Exception as e:
            print(f"Error accessing directory: {e}")
            return []

def analyze_leishmaniasis_files(files):
    """
    Analyze leishmaniasis files from file list
    """
    # Initialize counters
    case_counts = Counter()
    document_types = Counter()
    total_cases = 0
    
    # Patterns to identify different types and case numbers
    case_patterns = [
        (r'^(\d+)-case[s]?-', 'case_reports'),
        (r'^(\d+)-patient[s]?-', 'patient_reports'),
        (r'^(\d+)-images?-', 'image_studies'),
        (r'textbook|hunter|harrison|mandell|manson', 'textbooks'),
        (r'guideline|WHO|CDC', 'guidelines'),
        (r'good-multimodal', 'multimodal_resources')
    ]
    
    for filename in files:
        filename_lower = filename.lower()
        case_number = 0
        doc_type = 'other'
        
        # Check for case numbers and document types
        for pattern, dtype in case_patterns:
            match = re.search(pattern, filename_lower)
            if match:
                if dtype in ['textbooks', 'guidelines', 'multimodal_resources']:
                    doc_type = dtype
                    case_number = 0  # These don't contribute to case count
                else:
                    doc_type = dtype
                    if match.group(1).isdigit():
                        case_number = int(match.group(1))
                break
        
        # Special handling for single cases without explicit numbers
        if case_number == 0 and any(term in filename_lower for term in ['1-case', 'case report', 'case study']):
            case_number = 1
            doc_type = 'case_reports'
        
        # Count cases and document types
        if case_number > 0:
            case_counts[case_number] += 1
            total_cases += case_number
        
        document_types[doc_type] += 1
    
    return case_counts, document_types, total_cases, len(files)

def create_visualizations(case_counts, document_types, total_cases, total_files, output_dir=None):
    """
    Create comprehensive visualizations with improved text placement
    """
    # Create figure with subplots - increased figure size for better spacing
    fig = plt.figure(figsize=(24, 14))
    
    # 1. Case Distribution Bar Chart with improved text placement
    ax1 = plt.subplot(2, 3, 1)
    if case_counts:
        cases = sorted(case_counts.keys())
        counts = [case_counts[c] for c in cases]
        
        # Limit to top 10 categories to avoid overcrowding
        if len(cases) > 10:
            sorted_items = sorted(zip(cases, counts), key=lambda x: x[1], reverse=True)[:10]
            cases, counts = zip(*sorted_items)
            cases, counts = list(cases), list(counts)
        
        bars = ax1.bar([f"{c}" for c in cases], counts, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(cases))))
        ax1.set_title('Distribution of Case Reports\nby Number of Cases', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Number of Cases per Document', fontsize=10)
        ax1.set_ylabel('Number of Documents', fontsize=10)
        
        # Add value labels on bars with better positioning
        max_height = max(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            # Only show labels for bars with sufficient height
            if height > max_height * 0.05:
                ax1.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.01,
                        f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Rotate x-axis labels if there are many categories
        if len(cases) > 6:
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Document Types Pie Chart with better label positioning
    ax2 = plt.subplot(2, 3, 2)
    doc_labels = []
    doc_values = []
    doc_colors = plt.cm.Set2(np.linspace(0, 1, len(document_types)))
    
    for doc_type, count in document_types.items():
        # Shorten labels and use percentage for small slices
        short_label = doc_type.replace('_', ' ').title()
        if len(short_label) > 15:
            short_label = short_label[:12] + "..."
        doc_labels.append(short_label)
        doc_values.append(count)
    
    # Only show percentage labels for slices > 3%
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct > 3 else ''
    
    wedges, texts, autotexts = ax2.pie(doc_values, labels=None, autopct=autopct_format,
                                      colors=doc_colors, startangle=90, pctdistance=0.85)
    ax2.set_title('Distribution of Document Types', fontweight='bold', fontsize=12)
    
    # Create legend with counts
    legend_labels = [f"{label} ({count})" for label, count in zip(doc_labels, doc_values)]
    ax2.legend(wedges, legend_labels, title="Document Types", loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    
    # 3. Case Count Distribution (Histogram) with better binning
    ax3 = plt.subplot(2, 3, 3)
    if case_counts:
        case_numbers = []
        for case_num, freq in case_counts.items():
            case_numbers.extend([case_num] * freq)
        
        # Use appropriate number of bins
        num_bins = min(max(case_numbers), 20) if case_numbers else 1
        ax3.hist(case_numbers, bins=num_bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Histogram of Case Numbers', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Number of Cases in Document', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        
        # Improve x-axis labels for readability
        ax3.tick_params(axis='x', labelsize=9)
    
    # 4. Summary Statistics with better formatting
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    # Calculate statistics
    avg_cases_per_doc = total_cases / len([k for k, v in case_counts.items() for _ in range(v)]) if case_counts else 0
    max_cases = max(case_counts.keys()) if case_counts else 0
    
    summary_text = f"""LEISHMANIASIS LITERATURE SUMMARY

Total Documents: {total_files:,}
Total Cases: {total_cases:,}
Documents with Cases: {sum(case_counts.values()):,}
Reference Materials: {document_types.get('textbooks', 0) + document_types.get('guidelines', 0):,}

Case Statistics:
• Max cases in single document: {max_cases}
• Average cases per case document: {avg_cases_per_doc:.1f}
• Most common case count: {max(case_counts, key=case_counts.get) if case_counts else 'N/A'}

Document Breakdown:
• Case Reports: {document_types.get('case_reports', 0):,}
• Patient Reports: {document_types.get('patient_reports', 0):,}
• Image Studies: {document_types.get('image_studies', 0):,}
• Textbooks: {document_types.get('textbooks', 0):,}
• Guidelines: {document_types.get('guidelines', 0):,}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # 5. Cumulative Case Distribution with better spacing
    ax5 = plt.subplot(2, 3, 5)
    if case_counts:
        sorted_cases = sorted(case_counts.items())
        cumulative_docs = np.cumsum([count for _, count in sorted_cases])
        cumulative_cases = np.cumsum([case_num * count for case_num, count in sorted_cases])
        
        x_positions = range(len(sorted_cases))
        x_labels = [f"{case_num}" for case_num, _ in sorted_cases]
        
        ax5_twin = ax5.twinx()
        
        line1 = ax5.plot(x_positions, cumulative_docs, 'o-', color='blue', linewidth=2, 
                        label='Cumulative Documents', markersize=4)
        line2 = ax5_twin.plot(x_positions, cumulative_cases, 's-', color='red', linewidth=2, 
                             label='Cumulative Cases', markersize=4)
        
        ax5.set_xlabel('Case Number Category', fontsize=10)
        ax5.set_ylabel('Cumulative Documents', color='blue', fontsize=10)
        ax5_twin.set_ylabel('Cumulative Cases', color='red', fontsize=10)
        ax5.set_title('Cumulative Distribution', fontweight='bold', fontsize=12)
        
        # Set x-tick labels with rotation if needed
        ax5.set_xticks(x_positions)
        ax5.set_xticklabels(x_labels)
        if len(x_labels) > 8:
            plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        
        # Combine legends
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    # 6. Case Contribution Analysis with horizontal bars for better readability
    ax6 = plt.subplot(2, 3, 6)
    if case_counts:
        case_contributions = [(case_num, case_num * count, count) 
                             for case_num, count in case_counts.items()]
        case_contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top 10 to avoid overcrowding
        if len(case_contributions) > 10:
            case_contributions = case_contributions[:10]
        
        categories = [f"{cc[0]} case{'s' if cc[0] != 1 else ''}" for cc in case_contributions]
        contributions = [cc[1] for cc in case_contributions]
        
        bars = ax6.barh(categories, contributions, 
                       color=plt.cm.viridis(np.linspace(0, 1, len(categories))))
        ax6.set_title('Total Case Contribution\nby Category', fontweight='bold', fontsize=12)
        ax6.set_xlabel('Total Cases Contributed', fontsize=10)
        
        # Add value labels with better positioning
        max_contrib = max(contributions) if contributions else 1
        for i, (bar, contrib) in enumerate(zip(bars, contributions)):
            width = bar.get_width()
            ax6.text(width + max_contrib * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{contrib}', ha='left', va='center', fontweight='bold', fontsize=9)
        
        # Adjust margins for better label visibility
        ax6.margins(x=0.15)
    
    # Adjust layout with more padding
    plt.tight_layout(pad=3.0)
    
    # Save figure if output directory is provided
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"leishmaniasis_analysis_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {filepath}")
    
    return fig

def save_detailed_report(case_counts, document_types, total_cases, total_files, files, output_dir):
    """
    Save detailed text report to file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"leishmaniasis_report_{timestamp}.txt"
    report_filepath = os.path.join(output_dir, report_filename)
    
    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("LEISHMANIASIS LITERATURE ANALYSIS REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Documents Analyzed: {total_files}\n")
        f.write(f"Total Cases: {total_cases}\n\n")
        
        f.write("CASE DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        for case_num in sorted(case_counts.keys()):
            count = case_counts[case_num]
            total_cases_from_category = case_num * count
            f.write(f"{case_num:2d} case{'s' if case_num != 1 else ' '}: {count:3d} documents ({total_cases_from_category:3d} total cases)\n")
        
        f.write(f"\nDOCUMENT TYPES:\n")
        f.write("-" * 30 + "\n")
        for doc_type, count in sorted(document_types.items()):
            f.write(f"{doc_type.replace('_', ' ').title():20s}: {count:3d}\n")
        
        # Calculate some statistics
        avg_cases_per_doc = total_cases / len([k for k, v in case_counts.items() for _ in range(v)]) if case_counts else 0
        max_cases = max(case_counts.keys()) if case_counts else 0
        most_common = max(case_counts, key=case_counts.get) if case_counts else 'N/A'
        
        f.write(f"\nSTATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Max cases in single document: {max_cases}\n")
        f.write(f"Average cases per case document: {avg_cases_per_doc:.2f}\n")
        f.write(f"Most common case count: {most_common}\n")
        f.write(f"Documents with cases: {sum(case_counts.values())}\n")
        f.write(f"Reference materials: {document_types.get('textbooks', 0) + document_types.get('guidelines', 0)}\n")
    
    print(f"Detailed report saved to: {report_filepath}")
    return report_filepath

def save_csv_data(case_counts, document_types, output_dir):
    """
    Save data as CSV for further analysis
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Case distribution CSV
    case_df = pd.DataFrame([
        {'case_number': case_num, 'document_count': count, 'total_cases': case_num * count}
        for case_num, count in case_counts.items()
    ])
    case_csv_path = os.path.join(output_dir, f"case_distribution_{timestamp}.csv")
    case_df.to_csv(case_csv_path, index=False)
    
    # Document types CSV
    doc_df = pd.DataFrame([
        {'document_type': doc_type.replace('_', ' ').title(), 'count': count}
        for doc_type, count in document_types.items()
    ])
    doc_csv_path = os.path.join(output_dir, f"document_types_{timestamp}.csv")
    doc_df.to_csv(doc_csv_path, index=False)
    
    print(f"CSV data saved to:")
    print(f"  - {case_csv_path}")
    print(f"  - {doc_csv_path}")
    
    return case_csv_path, doc_csv_path

# Main execution function
def analyze_and_visualize(directory_path="data/standard", save_outputs=True):
    """
    Main function to analyze files and create visualizations
    Args:
        directory_path: Path to directory containing leishmaniasis files
        save_outputs: Whether to save outputs to files
    """
    print(f"Analyzing Leishmaniasis literature files in '{directory_path}'...")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get files from directory
    files = get_files_from_directory(directory_path)
    
    if not files:
        print("No files found or unable to access directory")
        return None
    
    # Analyze files
    case_counts, document_types, total_cases, total_files = analyze_leishmaniasis_files(files)
    
    print(f"Analysis complete!")
    print(f"Found {total_files} files with {total_cases} total cases")
    
    # Create visualizations
    output_dir = script_dir if save_outputs else None
    fig = create_visualizations(case_counts, document_types, total_cases, total_files, output_dir)
    
    # Save additional outputs if requested
    if save_outputs:
        # Save detailed report
        save_detailed_report(case_counts, document_types, total_cases, total_files, files, script_dir)
        
        # Save CSV data
        save_csv_data(case_counts, document_types, script_dir)
    
    # Display the plot
    plt.show()
    
    # Print detailed breakdown
    print("\n" + "="*50)
    print("DETAILED BREAKDOWN:")
    print("="*50)
    
    print(f"\nCase Distribution:")
    for case_num in sorted(case_counts.keys()):
        count = case_counts[case_num]
        total_cases_from_category = case_num * count
        print(f"  • {case_num} case{'s' if case_num != 1 else ''}: {count} documents ({total_cases_from_category} total cases)")
    
    print(f"\nDocument Types:")
    for doc_type, count in document_types.items():
        print(f"  • {doc_type.replace('_', ' ').title()}: {count}")
    
    if save_outputs:
        print(f"\nOutput files saved to: {script_dir}")
    
    # Return detailed results
    return {
        'case_counts': dict(case_counts),
        'document_types': dict(document_types),
        'total_cases': total_cases,
        'total_files': total_files,
        'files': files,
        'output_directory': script_dir if save_outputs else None
    }

# ============================================================================
# USAGE INSTRUCTIONS:
# ============================================================================
# Simply run one of these commands:

# 1. For your default directory with file outputs:
results = analyze_and_visualize("data/standard")

# 2. For a different directory:
# results = analyze_and_visualize("path/to/your/directory")

# 3. Without saving files (display only):
# results = analyze_and_visualize("data/standard", save_outputs=False)

# ============================================================================
# READY TO RUN - UNCOMMENT THE LINE BELOW:
# ============================================================================

results = analyze_and_visualize("data/standard")