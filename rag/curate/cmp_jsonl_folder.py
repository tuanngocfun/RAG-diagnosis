#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# Paths
extract_dir = Path("/home/students/Leishmania/kaggle/working2/extract")
jsonl_dir = Path("/home/students/Leishmania/kaggle/working2/rag_knowledge_base/qa/jsonl")

# 1. Get folder names from extract_dir
folder_names = sorted([f.name for f in extract_dir.iterdir() if f.is_dir()])

# 2. Get JSONL base filenames (without extension)
jsonl_files = sorted([f.stem for f in jsonl_dir.glob("*.jsonl")])

# 3. Find missing JSONLs for existing folders
missing_jsonl = [f for f in folder_names if f not in jsonl_files]

# 4. Find JSONLs that don't have corresponding folders
orphan_jsonl = [f for f in jsonl_files if f not in folder_names]

# 5. Print results
print("=== Missing JSONL files (folders exist, but JSONL missing) ===")
if missing_jsonl:
    for name in missing_jsonl:
        print(f"- {name}")
else:
    print("✅ None")

print("\n=== Orphan JSONL files (JSONL exists, but folder missing) ===")
if orphan_jsonl:
    for name in orphan_jsonl:
        print(f"- {name}")
else:
    print("✅ None")

# 6. Summary stats
print("\n=== Summary ===")
print(f"Total folders in extract/: {len(folder_names)}")
print(f"Total JSONL files:         {len(jsonl_files)}")
print(f"Missing JSONLs:            {len(missing_jsonl)}")
print(f"Orphan JSONLs:             {len(orphan_jsonl)}")
