#!/usr/bin/env python3
"""
Extract and analyze leishmaniasis-specific images from MultiCaRe dataset.

This script:
1. Loads leishmaniasis_matched_cases.csv (output from multicare_counts.py)
2. Filters captions_and_labels.csv to find images for those cases
3. Generates comprehensive statistics by image_type, image_subtype
4. Optionally extracts images to a structured folder

Usage:
    python extract_leishmaniasis_images.py --stats-only    # Just show statistics
    python extract_leishmaniasis_images.py --extract       # Extract images too
"""

from pathlib import Path
import pandas as pd
import zipfile
import json
import argparse
from collections import defaultdict
import io


ROOT = Path("./whole_multicare_dataset")


def load_leishmaniasis_cases() -> pd.DataFrame:
    """Load the matched leishmaniasis cases from multicare_counts.py output."""
    cases_path = ROOT / "leishmaniasis_matched_cases.csv"
    df = pd.read_csv(cases_path)
    print(f"✓ Loaded {len(df)} leishmaniasis cases from {cases_path.name}")
    return df


def load_captions_and_labels() -> pd.DataFrame:
    """Load the full captions and labels dataset."""
    labels_path = ROOT / "captions_and_labels.csv"
    print(f"  Loading {labels_path.name} (this may take a moment)...")
    df = pd.read_csv(labels_path)
    print(f"✓ Loaded {len(df)} image records")
    return df


def filter_leishmaniasis_images(
    leish_cases: pd.DataFrame,
    all_images: pd.DataFrame
) -> pd.DataFrame:
    """Filter images to only those belonging to leishmaniasis cases."""
    # patient_id in captions_and_labels.csv format: PMC10010164_01
    # case_id in leishmaniasis_matched_cases.csv format: PMC10010164_01
    leish_patient_ids = set(leish_cases['case_id'].unique())

    # Filter images where patient_id matches our leishmaniasis cases
    leish_images = all_images[all_images['patient_id'].isin(leish_patient_ids)].copy()
    print(f"✓ Found {len(leish_images)} images for {len(leish_patient_ids)} leishmaniasis cases")
    return leish_images


def analyze_image_distribution(leish_images: pd.DataFrame) -> dict:
    """Generate comprehensive statistics about leishmaniasis images."""
    stats = {}

    # Overall stats
    stats['total_images'] = len(leish_images)
    stats['unique_cases'] = leish_images['patient_id'].nunique()
    stats['unique_articles'] = leish_images['patient_id'].str.split('_').str[0].nunique()

    # By image type (primary modality)
    stats['by_image_type'] = leish_images['image_type'].value_counts().to_dict()

    # By image subtype (detailed modality)
    stats['by_image_subtype'] = leish_images['image_subtype'].value_counts().to_dict()

    # Cross-tabulation: image_type x image_subtype
    cross_tab = pd.crosstab(leish_images['image_type'], leish_images['image_subtype'])
    stats['type_subtype_crosstab'] = cross_tab.to_dict()

    # By license
    stats['by_license'] = leish_images['license'].value_counts().to_dict()

    # Has radiology region (only for radiology images)
    radiology_mask = leish_images['image_type'] == 'radiology'
    if radiology_mask.any():
        stats['radiology_regions'] = leish_images.loc[radiology_mask, 'radiology_region'].value_counts().to_dict()
        stats['radiology_views'] = leish_images.loc[radiology_mask, 'radiology_view'].value_counts().to_dict()

    # Caption length distribution
    caption_lengths = leish_images['caption'].fillna('').str.len()
    stats['caption_stats'] = {
        'mean': float(caption_lengths.mean()),
        'median': float(caption_lengths.median()),
        'min': int(caption_lengths.min()),
        'max': int(caption_lengths.max())
    }

    # File size distribution (in KB)
    file_sizes_kb = leish_images['file_size'] / 1024
    stats['file_size_kb_stats'] = {
        'mean': float(file_sizes_kb.mean()),
        'median': float(file_sizes_kb.median()),
        'total_mb': float(file_sizes_kb.sum() / 1024)
    }

    return stats


def print_statistics(stats: dict):
    """Pretty print the statistics."""
    print("\n" + "=" * 60)
    print("📊 LEISHMANIASIS IMAGE STATISTICS")
    print("=" * 60)

    print(f"\n📌 Overall:")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Unique cases: {stats['unique_cases']}")
    print(f"   Unique articles: {stats['unique_articles']}")
    print(f"   Total size: {stats['file_size_kb_stats']['total_mb']:.1f} MB")

    print(f"\n📷 By Image Type (Primary Modality):")
    for img_type, count in sorted(stats['by_image_type'].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total_images']
        bar = "█" * int(pct / 5)
        print(f"   {img_type:25s} {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n🔬 By Image Subtype (Top 15):")
    subtypes = sorted(stats['by_image_subtype'].items(), key=lambda x: -x[1])[:15]
    for subtype, count in subtypes:
        pct = 100 * count / stats['total_images']
        print(f"   {subtype:30s} {count:4d} ({pct:5.1f}%)")

    print(f"\n📄 By License:")
    for license_type, count in sorted(stats['by_license'].items(), key=lambda x: -x[1]):
        print(f"   {license_type:20s} {count:4d}")

    if 'radiology_regions' in stats:
        print(f"\n🏥 Radiology Anatomical Regions:")
        for region, count in sorted(stats['radiology_regions'].items(), key=lambda x: -x[1]):
            print(f"   {region:20s} {count:4d}")

    print(f"\n📝 Caption Statistics:")
    cs = stats['caption_stats']
    print(f"   Mean length: {cs['mean']:.0f} chars")
    print(f"   Median length: {cs['median']:.0f} chars")
    print(f"   Range: {cs['min']} - {cs['max']} chars")


def get_zip_for_file(filename: str) -> str:
    """Determine which zip file contains a given image."""
    # Filename format: PMC10000323_jbsr-107-1-3012-g3_undivided_1_1.webp
    # Path inside zip: PMC1/PMC10/PMC10000323_jbsr-107-1-3012-g3_undivided_1_1.webp
    # Zip file: PMC1.zip (based on first 4 chars)
    pmc_prefix = filename[:4]  # e.g., "PMC1" or "PMC9"
    return f"{pmc_prefix}.zip"


def extract_images(leish_images: pd.DataFrame, output_dir: Path):
    """Extract leishmaniasis images from zip files to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

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
                # Construct path inside zip
                filename = row['file']
                pmc_prefix = filename[:4]
                pmc_folder = filename[:5]
                inner_path = f"{pmc_prefix}/{pmc_folder}/{filename}"

                # Create output structure by case
                case_dir = output_dir / row['patient_id']
                case_dir.mkdir(exist_ok=True)

                try:
                    # Extract image
                    with zf.open(inner_path) as src:
                        img_data = src.read()
                        (case_dir / filename).write_bytes(img_data)

                    # Write metadata alongside image
                    metadata = {
                        'file_id': row['file_id'],
                        'file': row['file'],
                        'patient_id': row['patient_id'],
                        'caption': row['caption'],
                        'image_type': row['image_type'],
                        'image_subtype': row['image_subtype'],
                        'license': row['license'],
                        'labels_supervised': row.get('ml_labels_for_supervised_classification', ''),
                        'labels_semisupervised': row.get('gt_labels_for_semisupervised_classification', '')
                    }
                    meta_path = case_dir / f"{Path(filename).stem}_metadata.json"
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    extracted += 1

                except KeyError:
                    print(f"      ⚠ {inner_path} not found in zip")
                    errors += 1

    print(f"\n✓ Extraction complete: {extracted} images, {errors} errors")
    return extracted, errors


def save_statistics(stats: dict, leish_images: pd.DataFrame, output_dir: Path):
    """Save statistics and filtered image list."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save stats as JSON
    stats_path = output_dir / "leishmaniasis_image_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")

    # Save filtered image list for further processing
    images_path = output_dir / "leishmaniasis_images.csv"
    leish_images.to_csv(images_path, index=False)
    print(f"✓ Saved image list to {images_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract and analyze leishmaniasis images")
    parser.add_argument('--stats-only', action='store_true', help="Only show statistics, don't extract")
    parser.add_argument('--extract', action='store_true', help="Extract images to output folder")
    parser.add_argument('--output', type=Path, default=Path("./leishmaniasis_images"),
                        help="Output directory for extracted images")
    args = parser.parse_args()

    # Default to stats-only if neither flag is set
    if not args.stats_only and not args.extract:
        args.stats_only = True

    # Load data
    print("\n🔄 Loading data...")
    leish_cases = load_leishmaniasis_cases()
    all_images = load_captions_and_labels()

    # Filter to leishmaniasis images
    print("\n🔍 Filtering to leishmaniasis images...")
    leish_images = filter_leishmaniasis_images(leish_cases, all_images)

    # Analyze and print statistics
    print("\n📊 Analyzing image distribution...")
    stats = analyze_image_distribution(leish_images)
    print_statistics(stats)

    # Save statistics
    save_statistics(stats, leish_images, args.output)

    # Extract if requested
    if args.extract:
        extract_images(leish_images, args.output / "images")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
