#!/usr/bin/env python3
"""
Extract Validated Leishmaniasis Cases from MultiCaRe Dataset

This script extracts Leishmaniasis cases with:
1. Strict primary diagnosis verification
2. Deduplication
3. Image count from captions_and_labels.csv

Output: leishmaniasis_validated_cases.csv
"""

import duckdb
import pandas as pd
from pathlib import Path

ROOT = Path("./whole_multicare_dataset")
OUTPUT_CSV = ROOT / "leishmaniasis_validated_cases.csv"
OUTPUT_JSONL = ROOT / "leishmaniasis_validated_cases.jsonl"

def main():
    con = duckdb.connect()
    con.execute("INSTALL parquet; LOAD parquet;")
    
    meta_path = ROOT / "metadata.parquet"
    cases_path = ROOT / "cases.parquet"
    captions_path = ROOT / "captions_and_labels.csv"
    
    print("=" * 60)
    print("LEISHMANIASIS CASE EXTRACTION (VALIDATED)")
    print("=" * 60)
    
    # =========================================================================
    # STEP 1: Find articles with Leishmaniasis in metadata
    # =========================================================================
    print("\n[Step 1] Searching metadata for Leishmaniasis articles...")
    
    q_meta = f"""
    SELECT DISTINCT
        article_id::VARCHAR AS article_id,
        article_metadata.title::VARCHAR AS title,
        CAST(article_metadata.mesh_terms AS VARCHAR) AS mesh_terms,
        CAST(article_metadata.major_mesh_terms AS VARCHAR) AS major_mesh_terms,
        CAST(article_metadata.keywords AS VARCHAR) AS keywords
    FROM read_parquet('{meta_path.as_posix()}')
    WHERE 
        LOWER(article_metadata.title::VARCHAR) LIKE '%leishmaniasis%'
        OR LOWER(article_metadata.title::VARCHAR) LIKE '%leishmania%'
        OR LOWER(article_metadata.title::VARCHAR) LIKE '%kala-azar%'
        OR LOWER(article_metadata.title::VARCHAR) LIKE '%kala azar%'
        OR LOWER(CAST(article_metadata.mesh_terms AS VARCHAR)) LIKE '%leishmaniasis%'
        OR LOWER(CAST(article_metadata.mesh_terms AS VARCHAR)) LIKE '%leishmania%'
        OR LOWER(CAST(article_metadata.keywords AS VARCHAR)) LIKE '%leishmaniasis%'
        OR LOWER(CAST(article_metadata.keywords AS VARCHAR)) LIKE '%leishmania%'
    """
    
    meta_df = con.execute(q_meta).df()
    meta_lookup = {row['article_id']: row['title'] for _, row in meta_df.iterrows()}
    meta_article_ids = set(meta_df['article_id'].unique())
    print(f"   Found {len(meta_article_ids)} articles with Leishmaniasis in title/MeSH/keywords")
    
    # =========================================================================
    # STEP 2: Load all cases
    # =========================================================================
    print("\n[Step 2] Loading all cases from cases.parquet...")
    
    q_cases = f"""
    SELECT 
        rp.article_id::VARCHAR AS article_id,
        c.case_id::VARCHAR AS case_id,
        c.case_text::VARCHAR AS case_text,
        c.age AS age,
        c.gender::VARCHAR AS gender
    FROM read_parquet('{cases_path.as_posix()}') rp
    CROSS JOIN UNNEST(rp.cases) AS t(c)
    """
    
    all_cases = con.execute(q_cases).df()
    print(f"   Total cases in dataset: {len(all_cases)}")
    
    # =========================================================================
    # STEP 3: Filter by case_text containing leishmaniasis
    # =========================================================================
    print("\n[Step 3] Filtering cases by case_text content...")
    
    leish_keywords_primary = [
        'leishmaniasis', 'leishmania', 'kala-azar', 'kala azar', 
        'amastigote', 'ld bodies', 'ldck'
    ]
    
    def has_leish(text):
        if pd.isna(text):
            return False
        text_lower = text.lower()
        return any(kw in text_lower for kw in leish_keywords_primary)
    
    cases_with_text_leish = all_cases[all_cases['case_text'].apply(has_leish)]
    text_article_ids = set(cases_with_text_leish['article_id'].unique())
    print(f"   Cases with Leishmaniasis mentioned in text: {len(cases_with_text_leish)}")
    
    # =========================================================================
    # STEP 4: Combine and get all candidate cases
    # =========================================================================
    print("\n[Step 4] Combining article sources...")
    all_leish_article_ids = meta_article_ids.union(text_article_ids)
    print(f"   Total unique article IDs: {len(all_leish_article_ids)}")
    
    confirmed_leish_cases = all_cases[all_cases['article_id'].isin(all_leish_article_ids)].copy()
    print(f"   Total candidate cases: {len(confirmed_leish_cases)}")
    
    # =========================================================================
    # STEP 5: Validate PRIMARY diagnosis
    # =========================================================================
    print("\n[Step 5] Validating PRIMARY diagnosis...")
    
    # Strong indicators of primary leishmaniasis
    primary_patterns = [
        'diagnosed with leishmaniasis',
        'diagnosis of leishmaniasis',
        'confirmed leishmaniasis',
        'visceral leishmaniasis',
        'cutaneous leishmaniasis',
        'mucocutaneous leishmaniasis',
        'leishmaniasis was confirmed',
        'leishmania donovani',
        'leishmania infantum',
        'leishmania tropica',
        'leishmania major',
        'leishmania braziliensis',
        'leishmania mexicana',
        'leishmania amazonensis',
        'kala-azar',
        'amastigote',
        'ld bodies',
        'post-kala-azar dermal leishmaniasis',
        'pkdl'
    ]
    
    # Patterns that indicate OTHER diseases (leishmaniasis might be differential)
    other_disease_patterns = [
        'histoplasmosis', 'histoplasma', 
        'vexas', 
        'neurofibromatosis', 'nf1',
        'amyotrophic lateral sclerosis', 'als', 'motoneuron',
        'bronchogenic carcinoma',
        'rheumatic fever',
        'hodgkin lymphoma',
        'non-hodgkin',
        'multiple sclerosis',
        'psoriasis',
        'septorhinoplasty',
        'coronary artery disease',
        'osteoblastoma',
        'ecmo'
    ]
    
    def is_primary_leish(row):
        text = row['case_text']
        article_id = row['article_id']
        
        # If article title explicitly mentions leishmaniasis, high confidence
        article_in_meta = article_id in meta_article_ids
        
        if pd.isna(text) or len(str(text).strip()) < 50:
            return article_in_meta
        
        text_lower = text.lower()
        
        # Check for strong primary indicators
        has_primary = any(p in text_lower for p in primary_patterns)
        
        # Check if this is clearly about another disease
        has_other_dominant = any(p in text_lower for p in other_disease_patterns)
        
        # Decision logic:
        # 1. If has primary leishmaniasis terms and NO other dominant disease → Yes
        # 2. If article title says leishmaniasis AND no other dominant disease → Yes
        # 3. Otherwise → No
        
        if has_primary and not has_other_dominant:
            return True
        elif article_in_meta and not has_other_dominant:
            # Article metadata says leishmaniasis, case text doesn't contradict
            return True
        
        return False
    
    confirmed_leish_cases['is_primary'] = confirmed_leish_cases.apply(is_primary_leish, axis=1)
    primary_cases = confirmed_leish_cases[confirmed_leish_cases['is_primary']].copy()
    print(f"   Cases with PRIMARY leishmaniasis: {len(primary_cases)}")
    
    # =========================================================================
    # STEP 6: Deduplicate
    # =========================================================================
    print("\n[Step 6] Deduplicating...")
    primary_cases = primary_cases.drop_duplicates(subset=['case_id'])
    print(f"   After deduplication: {len(primary_cases)} unique cases")
    
    # =========================================================================
    # STEP 7: Add image counts
    # =========================================================================
    print("\n[Step 7] Adding image counts from captions_and_labels.csv...")
    
    captions_df = pd.read_csv(captions_path, usecols=['patient_id'])
    image_counts = captions_df['patient_id'].value_counts().to_dict()
    
    primary_cases['n_images'] = primary_cases['case_id'].map(
        lambda x: image_counts.get(x, 0)
    )
    
    total_images = primary_cases['n_images'].sum()
    cases_with_images = (primary_cases['n_images'] > 0).sum()
    print(f"   Total images: {total_images}")
    print(f"   Cases with images: {cases_with_images}/{len(primary_cases)}")
    
    # =========================================================================
    # STEP 8: Add article titles
    # =========================================================================
    primary_cases['article_title'] = primary_cases['article_id'].map(
        lambda x: meta_lookup.get(x, '')
    )
    
    # =========================================================================
    # STEP 9: Classify leishmaniasis type
    # =========================================================================
    print("\n[Step 8] Classifying leishmaniasis types...")
    
    def classify_leish_type(text, title):
        combined = (str(text) + " " + str(title)).lower()
        
        if 'visceral' in combined or 'kala-azar' in combined or 'kala azar' in combined:
            return 'Visceral'
        elif 'mucocutaneous' in combined:
            return 'Mucocutaneous'
        elif 'cutaneous' in combined:
            return 'Cutaneous'
        elif 'pkdl' in combined or 'post-kala-azar' in combined:
            return 'PKDL'
        else:
            return 'Unspecified'
    
    primary_cases['leish_type'] = primary_cases.apply(
        lambda r: classify_leish_type(r['case_text'], r['article_title']), axis=1
    )
    
    type_counts = primary_cases['leish_type'].value_counts()
    print("   Type distribution:")
    for t, c in type_counts.items():
        print(f"     {t}: {c}")
    
    # =========================================================================
    # STEP 10: Save outputs
    # =========================================================================
    print("\n[Step 9] Saving outputs...")
    
    # CSV output (minimal)
    csv_output = primary_cases[['case_id', 'article_id', 'n_images', 'leish_type']].copy()
    csv_output.to_csv(OUTPUT_CSV, index=False)
    print(f"   Saved {OUTPUT_CSV}")
    
    # JSONL output (full data)
    import json
    with open(OUTPUT_JSONL, 'w') as f:
        for _, row in primary_cases.iterrows():
            record = {
                'case_id': row['case_id'],
                'article_id': row['article_id'],
                'article_title': row['article_title'],
                'case_text': row['case_text'] if pd.notna(row['case_text']) else '',
                'age': row['age'] if pd.notna(row['age']) else None,
                'gender': row['gender'] if pd.notna(row['gender']) else None,
                'n_images': int(row['n_images']),
                'leish_type': row['leish_type']
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"   Saved {OUTPUT_JSONL}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total validated Leishmaniasis cases: {len(primary_cases)}")
    print(f"Cases with images: {cases_with_images}")
    print(f"Total images: {total_images}")
    print(f"\nCompared to original: leishmaniasis_matched_cases.csv had 407 cases")
    print(f"Reduction: {407 - len(primary_cases)} false positives removed")
    
    return primary_cases


if __name__ == "__main__":
    main()
