#!/usr/bin/env python3
"""
Build a complete MULTIMODAL leishmaniasis dataset with:
- Clinical case text (from cases.parquet)
- Article abstracts (from abstracts.parquet)  
- Images (from PMC*.zip)
- Captions & labels (from captions_and_labels.csv)

This creates a complete multimodal dataset suitable for RAG/fine-tuning.

Usage:
    python build_multimodal_dataset.py
"""

from pathlib import Path
import pandas as pd
import duckdb
import zipfile
import json
from collections import defaultdict


ROOT = Path("./whole_multicare_dataset")
OUTPUT = Path("./leishmaniasis_multimodal")


def load_leishmaniasis_cases_csv() -> pd.DataFrame:
    """Load the matched leishmaniasis cases from multicare_counts.py output."""
    cases_path = ROOT / "leishmaniasis_matched_cases.csv"
    df = pd.read_csv(cases_path)
    print(f"✓ Loaded {len(df)} leishmaniasis case IDs from {cases_path.name}")
    return df


def load_case_texts(leish_case_ids: set) -> pd.DataFrame:
    """Load full case text from cases.parquet for leishmaniasis cases."""
    con = duckdb.connect()
    con.execute("INSTALL parquet; LOAD parquet;")
    
    cases_path = ROOT / "cases.parquet"
    
    # Extract all cases from the nested structure
    q = f"""
    SELECT 
        rp.article_id::VARCHAR AS article_id,
        c.case_id::VARCHAR AS case_id,
        c.case_text::VARCHAR AS case_text,
        c.age AS age,
        c.gender::VARCHAR AS gender
    FROM read_parquet('{cases_path.as_posix()}') rp
    CROSS JOIN UNNEST(rp.cases) AS t(c)
    """
    all_cases = con.execute(q).df()
    
    # Filter to leishmaniasis cases
    leish_cases = all_cases[all_cases['case_id'].isin(leish_case_ids)]
    print(f"✓ Loaded case text for {len(leish_cases)} cases from cases.parquet")
    
    return leish_cases


def load_abstracts(article_ids: set) -> pd.DataFrame:
    """Load abstracts from abstracts.parquet for leishmaniasis articles."""
    con = duckdb.connect()
    con.execute("INSTALL parquet; LOAD parquet;")
    
    abs_path = ROOT / "abstracts.parquet"
    
    q = f"""
    SELECT 
        article_id::VARCHAR AS article_id,
        abstract::VARCHAR AS abstract
    FROM read_parquet('{abs_path.as_posix()}')
    """
    all_abstracts = con.execute(q).df()
    
    # Filter to leishmaniasis articles
    leish_abstracts = all_abstracts[all_abstracts['article_id'].isin(article_ids)]
    print(f"✓ Loaded abstracts for {len(leish_abstracts)} articles from abstracts.parquet")
    
    return leish_abstracts


def load_metadata(article_ids: set) -> pd.DataFrame:
    """Load article metadata from metadata.parquet."""
    con = duckdb.connect()
    con.execute("INSTALL parquet; LOAD parquet;")
    
    meta_path = ROOT / "metadata.parquet"
    
    q = f"""
    SELECT 
        article_id::VARCHAR AS article_id,
        article_metadata.title::VARCHAR AS title,
        article_metadata.journal::VARCHAR AS journal,
        article_metadata.year AS year,
        article_metadata.doi::VARCHAR AS doi,
        article_metadata.pmid::VARCHAR AS pmid,
        article_metadata.license::VARCHAR AS license,
        CAST(article_metadata.mesh_terms AS VARCHAR) AS mesh_terms,
        CAST(article_metadata.major_mesh_terms AS VARCHAR) AS major_mesh_terms,
        CAST(article_metadata.keywords AS VARCHAR) AS keywords,
        article_metadata.link::VARCHAR AS link
    FROM read_parquet('{meta_path.as_posix()}')
    """
    all_meta = con.execute(q).df()
    
    # Filter to leishmaniasis articles
    leish_meta = all_meta[all_meta['article_id'].isin(article_ids)]
    print(f"✓ Loaded metadata for {len(leish_meta)} articles from metadata.parquet")
    
    return leish_meta


def load_captions_and_labels() -> pd.DataFrame:
    """Load the full captions and labels dataset."""
    labels_path = ROOT / "captions_and_labels.csv"
    print(f"  Loading {labels_path.name}...")
    df = pd.read_csv(labels_path)
    print(f"✓ Loaded {len(df)} image records from captions_and_labels.csv")
    return df


def filter_leishmaniasis_images(leish_case_ids: set, all_images: pd.DataFrame) -> pd.DataFrame:
    """Filter images to only those belonging to leishmaniasis cases."""
    leish_images = all_images[all_images['patient_id'].isin(leish_case_ids)].copy()
    print(f"✓ Found {len(leish_images)} images for leishmaniasis cases")
    return leish_images


def get_zip_for_file(filename: str) -> str:
    """Determine which zip file contains a given image."""
    pmc_prefix = filename[:4]
    return f"{pmc_prefix}.zip"


def extract_images(leish_images: pd.DataFrame, output_dir: Path) -> tuple[int, int]:
    """Extract leishmaniasis images from zip files to output directory."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Group images by zip file
    zip_groups = defaultdict(list)
    for _, row in leish_images.iterrows():
        zip_name = get_zip_for_file(row['file'])
        zip_groups[zip_name].append(row)

    print(f"\n📦 Extracting images from {len(zip_groups)} zip files...")

    extracted = 0
    errors = 0

    for zip_name, rows in zip_groups.items():
        zip_path = ROOT / zip_name
        if not zip_path.exists():
            print(f"   ⚠ {zip_name} not found, skipping {len(rows)} images")
            errors += len(rows)
            continue

        print(f"   Processing {zip_name} ({len(rows)} images)...")

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for row in rows:
                filename = row['file']
                pmc_prefix = filename[:4]
                pmc_folder = filename[:5]
                inner_path = f"{pmc_prefix}/{pmc_folder}/{filename}"

                case_dir = images_dir / row['patient_id']
                case_dir.mkdir(exist_ok=True)

                try:
                    with zf.open(inner_path) as src:
                        img_data = src.read()
                        (case_dir / filename).write_bytes(img_data)
                    extracted += 1
                except KeyError:
                    print(f"      ⚠ {inner_path} not found in zip")
                    errors += 1

    print(f"✓ Extracted {extracted} images ({errors} errors)")
    return extracted, errors


def build_multimodal_records(
    cases_df: pd.DataFrame,
    abstracts_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    images_df: pd.DataFrame,
    output_dir: Path
) -> list[dict]:
    """Build complete multimodal records combining all data sources."""
    
    records = []
    
    # Create lookups
    abstracts_lookup = dict(zip(abstracts_df['article_id'], abstracts_df['abstract']))
    meta_lookup = metadata_df.set_index('article_id').to_dict('index')
    
    # Group images by case_id
    images_by_case = images_df.groupby('patient_id').apply(
        lambda x: x.to_dict('records')
    ).to_dict()
    
    for _, case_row in cases_df.iterrows():
        case_id = case_row['case_id']
        article_id = case_row['article_id']
        
        record = {
            # Case information
            'case_id': case_id,
            'article_id': article_id,
            'case_text': case_row['case_text'],
            'age': case_row['age'] if pd.notna(case_row['age']) else None,
            'gender': case_row['gender'],
            
            # Abstract
            'abstract': abstracts_lookup.get(article_id, ''),
            
            # Article metadata
            'title': meta_lookup.get(article_id, {}).get('title', ''),
            'journal': meta_lookup.get(article_id, {}).get('journal', ''),
            'year': meta_lookup.get(article_id, {}).get('year'),
            'doi': meta_lookup.get(article_id, {}).get('doi', ''),
            'pmid': meta_lookup.get(article_id, {}).get('pmid', ''),
            'license': meta_lookup.get(article_id, {}).get('license', ''),
            'mesh_terms': meta_lookup.get(article_id, {}).get('mesh_terms', ''),
            'major_mesh_terms': meta_lookup.get(article_id, {}).get('major_mesh_terms', ''),
            'keywords': meta_lookup.get(article_id, {}).get('keywords', ''),
            'link': meta_lookup.get(article_id, {}).get('link', ''),
            
            # Images with their captions and labels
            'images': []
        }
        
        # Add image information
        case_images = images_by_case.get(case_id, [])
        for img in case_images:
            record['images'].append({
                'file': img['file'],
                'file_id': img['file_id'],
                'caption': img['caption'],
                'image_type': img['image_type'],
                'image_subtype': img['image_subtype'],
                'radiology_region': img.get('radiology_region'),
                'radiology_view': img.get('radiology_view'),
                'labels_supervised': img.get('ml_labels_for_supervised_classification', ''),
                'labels_semisupervised': img.get('gt_labels_for_semisupervised_classification', '')
            })
        
        records.append(record)
    
    return records


def save_dataset(records: list[dict], output_dir: Path):
    """Save the complete multimodal dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL (one record per line, good for streaming)
    jsonl_path = output_dir / "leishmaniasis_multimodal.jsonl"
    with open(jsonl_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(records)} records to {jsonl_path.name}")
    
    # Also save as single JSON for easier viewing
    json_path = output_dir / "leishmaniasis_multimodal.json"
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved dataset to {json_path.name}")
    
    # Generate summary statistics
    stats = {
        'total_cases': len(records),
        'cases_with_images': sum(1 for r in records if r['images']),
        'total_images': sum(len(r['images']) for r in records),
        'cases_with_abstract': sum(1 for r in records if r['abstract']),
        'avg_case_text_length': sum(len(r['case_text'] or '') for r in records) / len(records),
        'image_type_distribution': {},
        'by_license': {}
    }
    
    for record in records:
        license_type = record['license'] or 'Unknown'
        stats['by_license'][license_type] = stats['by_license'].get(license_type, 0) + 1
        
        for img in record['images']:
            img_type = img['image_type'] or 'Unknown'
            stats['image_type_distribution'][img_type] = stats['image_type_distribution'].get(img_type, 0) + 1
    
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_path.name}")
    
    return stats


def print_summary(stats: dict):
    """Print a nice summary of the dataset."""
    print("\n" + "=" * 60)
    print("📊 LEISHMANIASIS MULTIMODAL DATASET SUMMARY")
    print("=" * 60)
    
    print(f"\n📌 Overall:")
    print(f"   Total cases: {stats['total_cases']}")
    print(f"   Cases with images: {stats['cases_with_images']}")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Cases with abstract: {stats['cases_with_abstract']}")
    print(f"   Avg case text length: {stats['avg_case_text_length']:.0f} chars")
    
    print(f"\n📷 Image Type Distribution:")
    for img_type, count in sorted(stats['image_type_distribution'].items(), key=lambda x: -x[1]):
        print(f"   {img_type:25s} {count:4d}")
    
    print(f"\n📄 By License:")
    for license_type, count in sorted(stats['by_license'].items(), key=lambda x: -x[1]):
        print(f"   {license_type:20s} {count:4d}")


def main():
    print("\n🔄 Building complete multimodal leishmaniasis dataset...\n")
    
    # Load leishmaniasis case IDs
    leish_cases_csv = load_leishmaniasis_cases_csv()
    leish_case_ids = set(leish_cases_csv['case_id'].unique())
    leish_article_ids = set(leish_cases_csv['article_id'].unique())
    
    # Load text data
    print("\n📄 Loading text data...")
    cases_df = load_case_texts(leish_case_ids)
    abstracts_df = load_abstracts(leish_article_ids)
    metadata_df = load_metadata(leish_article_ids)
    
    # Load image data
    print("\n🖼️ Loading image data...")
    all_images = load_captions_and_labels()
    leish_images = filter_leishmaniasis_images(leish_case_ids, all_images)
    
    # Build multimodal records
    print("\n🔧 Building multimodal records...")
    records = build_multimodal_records(
        cases_df, abstracts_df, metadata_df, leish_images, OUTPUT
    )
    
    # Extract images
    extract_images(leish_images, OUTPUT)
    
    # Save dataset
    print("\n💾 Saving dataset...")
    stats = save_dataset(records, OUTPUT)
    
    # Print summary
    print_summary(stats)
    
    print("\n✅ Complete multimodal dataset ready!")
    print(f"   Location: {OUTPUT.absolute()}/")
    print(f"   Main file: leishmaniasis_multimodal.jsonl")


if __name__ == "__main__":
    main()
