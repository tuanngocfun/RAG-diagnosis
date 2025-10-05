#!/usr/bin/env python3
"""
Verification script to confirm Gemini comparison CSV uses correct configuration.

This script validates that the Gemini delta CSV file uses values from the
text_size12k configuration consistently (not mixing in gemimi_rerank or MedGemma values).
"""

import pandas as pd
from pathlib import Path

def verify_gemini_comparison():
    """Verify that Gemini comparison CSV has correct values."""
    
    script_dir = Path(__file__).parent
    
    # Load source data
    main_csv = script_dir / 'Updated_RAG_experiment_comparison_with_MedGemma4B_standalone.csv'
    delta_csv = script_dir / 'Gemini_Pro_2_5___RAG_vs_Base__exact_deltas____.csv'
    
    print("🔍 Loading source data...")
    main_df = pd.read_csv(main_csv)
    delta_df = pd.read_csv(delta_csv)
    
    # Get expected values
    gemini_base = main_df[main_df['run'] == 'gem25-gemini-rpm-5 (missing)'].iloc[0]
    text_size12k = main_df[main_df['run'] == 'text_size12k, token1024 (missing)'].iloc[0]
    gemimi_rerank = main_df[main_df['run'] == 'gemimi_rerank_rag_test (all)'].iloc[0]
    medgemma_rtx = main_df[main_df['run'] == 'RTX3090 cross-enc MedCPT (final, all)'].iloc[0]
    
    print("\n" + "="*80)
    print("📊 SOURCE DATA VALUES")
    print("="*80)
    
    print("\n1️⃣  Gemini Baseline (gem25-gemini-rpm-5):")
    print(f"   F1: {gemini_base['avg_f1']:.10f}")
    print(f"   Faithfulness: {gemini_base['avg_faithfulness']:.10f}")
    print(f"   Correctness: {gemini_base['avg_correctness']:.10f}")
    print(f"   Completeness: {gemini_base['avg_completeness']:.10f}")
    
    print("\n2️⃣  Gemini text_size12k (SHOULD be used for comparison):")
    print(f"   F1: {text_size12k['avg_f1']:.10f}")
    print(f"   Faithfulness: {text_size12k['avg_faithfulness']:.10f}")
    print(f"   Correctness: {text_size12k['avg_correctness']:.10f}")
    print(f"   Completeness: {text_size12k['avg_completeness']:.10f}")
    
    print("\n3️⃣  Gemini gemimi_rerank (SHOULD NOT be mixed in):")
    print(f"   F1: {gemimi_rerank['avg_f1']:.10f}")
    print(f"   Faithfulness: {gemimi_rerank['avg_faithfulness']:.10f}")
    print(f"   Correctness: {gemimi_rerank['avg_correctness']:.10f}")
    print(f"   Completeness: {gemimi_rerank['avg_completeness']:.10f}")
    
    print("\n4️⃣  MedGemma RTX3090 (SHOULD NOT be used for Gemini comparison):")
    print(f"   F1: {medgemma_rtx['avg_f1']:.10f}")
    print(f"   Faithfulness: {medgemma_rtx['avg_faithfulness']:.10f}")
    print(f"   Correctness: {medgemma_rtx['avg_correctness']:.10f}")
    print(f"   Completeness: {medgemma_rtx['avg_completeness']:.10f}")
    
    # Verify delta CSV
    print("\n" + "="*80)
    print("✅ VERIFICATION: Delta CSV Values")
    print("="*80)
    
    errors = []
    warnings = []
    
    # Check each metric
    for metric in ['F1', 'Faithfulness', 'Correctness', 'Completeness']:
        delta_row = delta_df[delta_df['Metric'] == metric].iloc[0]
        
        # Get expected values
        if metric == 'F1':
            expected_base = gemini_base['avg_f1']
            expected_rag = text_size12k['avg_f1']
        elif metric == 'Faithfulness':
            expected_base = gemini_base['avg_faithfulness']
            expected_rag = text_size12k['avg_faithfulness']
        elif metric == 'Correctness':
            expected_base = gemini_base['avg_correctness']
            expected_rag = text_size12k['avg_correctness']
        elif metric == 'Completeness':
            expected_base = gemini_base['avg_completeness']
            expected_rag = text_size12k['avg_completeness']
        
        actual_base = float(delta_row['Gemini Base (no RAG)'])
        actual_rag = float(delta_row['Gemini Best (RAG)'])
        
        # Check if values match (with small tolerance for floating point)
        base_match = abs(actual_base - expected_base) < 1e-6
        rag_match = abs(actual_rag - expected_rag) < 1e-6
        
        # Check if using wrong config
        is_gemimi_rerank = False
        is_medgemma = False
        
        if metric == 'Faithfulness' and abs(actual_rag - medgemma_rtx['avg_faithfulness']) < 1e-6:
            is_medgemma = True
            errors.append(f"❌ {metric}: Using MedGemma RTX3090 value ({actual_rag:.4f}) instead of text_size12k ({expected_rag:.4f})")
        elif metric in ['Correctness', 'Completeness']:
            if metric == 'Correctness' and abs(actual_rag - gemimi_rerank['avg_correctness']) < 1e-6:
                is_gemimi_rerank = True
                errors.append(f"❌ {metric}: Using gemimi_rerank value ({actual_rag:.4f}) instead of text_size12k ({expected_rag:.4f})")
            elif metric == 'Completeness' and abs(actual_rag - gemimi_rerank['avg_completeness']) < 1e-6:
                is_gemimi_rerank = True
                errors.append(f"❌ {metric}: Using gemimi_rerank value ({actual_rag:.4f}) instead of text_size12k ({expected_rag:.4f})")
        
        if base_match and rag_match:
            print(f"✅ {metric:15s} - Baseline: {actual_base:.4f}, Best RAG: {actual_rag:.4f}")
        else:
            if not base_match:
                errors.append(f"❌ {metric} baseline mismatch: Expected {expected_base:.4f}, got {actual_base:.4f}")
            if not rag_match and not is_gemimi_rerank and not is_medgemma:
                errors.append(f"❌ {metric} RAG mismatch: Expected {expected_rag:.4f}, got {actual_rag:.4f}")
    
    # Calculate expected deltas
    print("\n" + "="*80)
    print("📈 DELTA VERIFICATION")
    print("="*80)
    
    for metric in ['F1', 'Faithfulness', 'Correctness', 'Completeness']:
        delta_row = delta_df[delta_df['Metric'] == metric].iloc[0]
        
        if metric == 'F1':
            expected_base = gemini_base['avg_f1']
            expected_rag = text_size12k['avg_f1']
        elif metric == 'Faithfulness':
            expected_base = gemini_base['avg_faithfulness']
            expected_rag = text_size12k['avg_faithfulness']
        elif metric == 'Correctness':
            expected_base = gemini_base['avg_correctness']
            expected_rag = text_size12k['avg_correctness']
        elif metric == 'Completeness':
            expected_base = gemini_base['avg_completeness']
            expected_rag = text_size12k['avg_completeness']
        
        expected_abs_delta = expected_rag - expected_base
        expected_rel_delta = (expected_abs_delta / expected_base) * 100 if expected_base != 0 else 0
        
        actual_abs_delta = float(delta_row['Absolute Δ (Best - Base)'])
        actual_rel_delta = float(delta_row['Relative Δ (%)'])
        
        abs_delta_match = abs(actual_abs_delta - expected_abs_delta) < 1e-4
        rel_delta_match = abs(actual_rel_delta - expected_rel_delta) < 0.5  # Allow 0.5% tolerance
        
        sign = "+" if expected_rel_delta >= 0 else ""
        
        if abs_delta_match and rel_delta_match:
            print(f"✅ {metric:15s} - Δ: {sign}{actual_rel_delta:+6.1f}%")
        else:
            if not abs_delta_match:
                errors.append(f"❌ {metric} absolute delta mismatch: Expected {expected_abs_delta:.6f}, got {actual_abs_delta:.6f}")
            if not rel_delta_match:
                errors.append(f"❌ {metric} relative delta mismatch: Expected {expected_rel_delta:.1f}%, got {actual_rel_delta:.1f}%")
    
    # Print results
    print("\n" + "="*80)
    print("🎯 VERIFICATION RESULTS")
    print("="*80)
    
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n✅ The Gemini comparison CSV correctly uses text_size12k configuration")
        print("✅ All baseline values match gem25-gemini-rpm-5")
        print("✅ All 'Best RAG' values match text_size12k")
        print("✅ Delta calculations are accurate")
        print("\n📊 Key Finding: Faithfulness delta is NEGATIVE (-10.9%)")
        print("   This correctly shows the trade-off: best F1 config sacrifices faithfulness")
        return True

if __name__ == "__main__":
    success = verify_gemini_comparison()
    exit(0 if success else 1)
