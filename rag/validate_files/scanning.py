import os

# Paths
qa_jsonl_dir = "kaggle/working2/rag_knowledge_base/qa/jsonl"
extract_dir = "kaggle/working2/extract"

# Step 1: Get all .jsonl filenames (without extension)
qa_files = {
    os.path.splitext(f)[0]
    for f in os.listdir(qa_jsonl_dir)
    if f.endswith(".jsonl")
}

# Step 2: Get all folder names from extract/
extract_folders = {
    f for f in os.listdir(extract_dir)
    if os.path.isdir(os.path.join(extract_dir, f))
}

# Step 3: Find missing files (folders with no corresponding .jsonl)
missing_files = sorted(extract_folders - qa_files)

# Step 4: Print results
print("=== Summary ===")
print(f"Total folders in extract/: {len(extract_folders)}")
print(f"Total JSONL files in qa/: {len(qa_files)}")
print(f"Missing JSONL files: {len(missing_files)}\n")

if missing_files:
    print("=== Missing JSONL files ===")
    for name in missing_files:
        print(name)
else:
    print("✅ All folders have corresponding JSONL files!")
