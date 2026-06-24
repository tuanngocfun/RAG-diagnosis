#!/usr/bin/env python3
"""Analyze quick test results: RAG (augmented) vs No-RAG."""
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict


def extract_diagnosis_type(answer):
    """Extract diagnosis type from structured answer."""
    match = re.search(r'\*\*Diagnosis Type:\*\*\s*([A-Z]+)', answer)
    return match.group(1) if match else 'N/A'


def extract_evidence_source(answer):
    """Extract evidence source from RAG answer (fixed to avoid overlap)."""
    match = re.search(r'\*\*Evidence Source:\*\*\s*(.+)', answer)
    if not match:
        return 'unknown'
    
    source = match.group(1).strip()
    # Check in order of specificity to avoid overlap
    if 'Retrieved cases only' in source:
        return 'retrieved_only'
    elif 'Retrieved + general' in source:
        return 'retrieved_plus'
    elif 'general knowledge' in source.lower():
        return 'retrieved_plus'
    
    return 'unknown'


def normalize(diagnosis):
    """Normalize diagnosis type to standard abbreviation."""
    if not diagnosis:
        return ''
    s = diagnosis.upper().strip()
    
    # Order matters: check PKDL first (contains 'KL')
    if 'PKDL' in s or 'POST-KALA-AZAR' in s or 'POST KALA AZAR' in s:
        return 'PKDL'
    if 'VL' in s or 'VISCERAL' in s:
        return 'VL'
    if 'CL' in s or 'CUTANEOUS' in s:
        return 'CL'
    if 'MCL' in s or 'MUCOCUTANEOUS' in s:
        return 'MCL'
    
    return s


def analyze_results(run_dir):
    """Analyze RAG vs NoRAG results."""
    
    # Load data
    rag_file = run_dir / 'rag_answers.jsonl'
    norag_file = run_dir / 'norag_answers.jsonl'
    
    if not rag_file.exists() or not norag_file.exists():
        print(f"❌ Error: Missing answer files in {run_dir}")
        return
    
    rag_answers = [json.loads(l) for l in rag_file.open()]
    norag_answers = [json.loads(l) for l in norag_file.open()]
    
    # Stats
    stats = {
        'total': len(rag_answers),
        'rag_correct': 0,
        'norag_correct': 0,
        'both_correct': 0,
        'both_wrong': 0,
        'rag_only': 0,
        'norag_only': 0,
        'evidence': defaultdict(int),
        'errors': []
    }
    
    # Header
    print('=' * 90)
    print('QUICK TEST RESULTS: RAG (augmented) vs No-RAG')
    print('=' * 90)
    print()
    
    # Table header
    print(f'{"Case ID":<22} | {"GT":<6} | {"RAG":<6} | {"NoRAG":<6} | RAG | NoRAG | Evidence')
    print('-' * 90)
    
    # Analyze each case
    for rag, norag in zip(rag_answers, norag_answers):
        gt = rag.get('ground_truth', {})
        gt_type = normalize(gt.get('diagnosis_type', '')) if gt else ''
        
        rag_type = normalize(extract_diagnosis_type(rag['answer']))
        norag_type = normalize(extract_diagnosis_type(norag['answer']))
        
        rag_match = rag_type == gt_type and gt_type != ''
        norag_match = norag_type == gt_type and gt_type != ''
        
        # Count matches
        if rag_match:
            stats['rag_correct'] += 1
        if norag_match:
            stats['norag_correct'] += 1
        
        # Detailed breakdown
        if rag_match and norag_match:
            stats['both_correct'] += 1
        elif not rag_match and not norag_match:
            stats['both_wrong'] += 1
        elif rag_match:
            stats['rag_only'] += 1
        else:
            stats['norag_only'] += 1
            # Track errors where NoRAG succeeded but RAG failed
            stats['errors'].append({
                'case_id': rag['case_id'],
                'gt': gt_type,
                'rag': rag_type,
                'norag': norag_type
            })
        
        # Evidence source
        evidence = extract_evidence_source(rag['answer'])
        stats['evidence'][evidence] += 1
        
        # Print row
        case_id_short = rag["case_id"][:20]
        evidence_short = evidence.replace('_', ' ')[:12]
        
        print(f'{case_id_short:<22} | {gt_type:<6} | {rag_type:<6} | {norag_type:<6} | '
              f'{"✓" if rag_match else "✗":3} | {"✓" if norag_match else "✗":5} | {evidence_short}')
    
    print('-' * 90)
    print()
    
    # Summary statistics
    total = stats['total']
    rag_acc = 100 * stats['rag_correct'] / total if total > 0 else 0
    norag_acc = 100 * stats['norag_correct'] / total if total > 0 else 0
    improvement = rag_acc - norag_acc
    
    print('📊 ACCURACY (Diagnosis Type Match):')
    print(f'  RAG (augmented):   {stats["rag_correct"]}/{total} = {rag_acc:.1f}%')
    print(f'  No-RAG:            {stats["norag_correct"]}/{total} = {norag_acc:.1f}%')
    print(f'  Improvement:       {improvement:+.1f}%')
    print()
    
    # Detailed breakdown
    print('🔍 DETAILED BREAKDOWN:')
    print(f'  Both correct:       {stats["both_correct"]} cases')
    print(f'  Both wrong:         {stats["both_wrong"]} cases')
    print(f'  RAG only correct:   {stats["rag_only"]} cases')
    print(f'  NoRAG only correct: {stats["norag_only"]} cases')
    print()
    
    # Evidence source analysis
    print('📚 EVIDENCE SOURCE (RAG):')
    total_with_source = sum(v for k, v in stats['evidence'].items() if k != 'unknown')
    
    for source in ['retrieved_only', 'retrieved_plus', 'unknown']:
        count = stats['evidence'][source]
        pct = 100 * count / total_with_source if total_with_source > 0 else 0
        label = source.replace('_', ' ').title()
        print(f'  {label:<27} {count:2} ({pct:.0f}%)')
    
    # Error analysis
    if stats['errors']:
        print()
        print('⚠️  CASES WHERE RAG FAILED BUT NoRAG SUCCEEDED:')
        for err in stats['errors']:
            print(f'  • {err["case_id"]:20} GT={err["gt"]:6} | RAG predicted {err["rag"]}, NoRAG predicted {err["norag"]}')
        print()
        print('  ⚡ Action: Review retrieved contexts for these cases to identify misleading information')
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze RAG vs NoRAG quick test results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_quick_test.py --run-dir runs/prompt_fix_quick_test_20260119_140804
  python analyze_quick_test.py  # Uses default run directory
        """
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        default='runs/prompt_fix_quick_test_20260119_140804',
        help='Path to run directory (default: runs/prompt_fix_quick_test_20260119_140804)'
    )
    
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    
    if not run_dir.exists():
        print(f"❌ Error: Run directory not found: {run_dir}")
        print(f"   Available runs:")
        runs_dir = Path('runs')
        if runs_dir.exists():
            for d in sorted(runs_dir.glob('*'), reverse=True)[:5]:
                print(f"   - {d.name}")
        return
    
    analyze_results(run_dir)


if __name__ == '__main__':
    main()