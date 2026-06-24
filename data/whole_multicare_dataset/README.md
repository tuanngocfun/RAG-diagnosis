# Whole MultiCaRe Dataset Ledger

This directory is a lightweight ledger for the local MultiCaRe/leishmaniasis
dataset used during thesis experiments. It is not the full raw dataset.

## Included

Small derived metadata files are included so experiments and presentation
provenance can be audited without committing the full raw corpus.

- `data_dictionary.csv`
- `leishmaniasis_*`
- `INCLUDED_SHA256SUMS.txt`
- `EXCLUDED_RAW_ARTIFACTS.tsv`

## Excluded Raw Artifacts

The following local source artifacts are intentionally not committed:

- `PMC*.zip`
- `*.parquet`
- `captions_and_labels.csv`

They are listed with byte sizes and source paths in
`EXCLUDED_RAW_ARTIFACTS.tsv`. Keep those files in local or external storage, or
publish them through a release/LFS flow only after an explicit data-governance
decision.

## Source

Local source directory:

`/home/ngocnt/Leishmaniasis_v3/data/whole_multicare_dataset`
