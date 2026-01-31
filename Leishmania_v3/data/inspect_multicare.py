from pathlib import Path
import duckdb
import pandas as pd

root = Path("./whole_multicare_dataset")

# 1) xem data_dictionary để biết schema/ý nghĩa cột
dd = pd.read_csv(root / "data_dictionary.csv")
print("\n=== data_dictionary.csv (top 30) ===")
print(dd.head(30))

# 2) dùng DuckDB để xem nhanh schema + vài dòng mẫu (không load full)
con = duckdb.connect()
con.execute("INSTALL parquet; LOAD parquet;")

def describe_parquet(p: Path, n=5):
    print(f"\n=== {p.name}: DESCRIBE ===")
    print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p.as_posix()}')").df())
    print(f"\n=== {p.name}: SAMPLE {n} ===")
    print(con.execute(f"SELECT * FROM read_parquet('{p.as_posix()}') LIMIT {n}").df())

describe_parquet(root / "cases.parquet", n=3)
describe_parquet(root / "metadata.parquet", n=3)
describe_parquet(root / "case_images.parquet", n=3)
describe_parquet(root / "abstracts.parquet", n=3)

# 3) csv captions + labels
lab = pd.read_csv(root / "captions_and_labels.csv")
print("\n=== captions_and_labels.csv columns ===")
print(list(lab.columns))
print("\n=== captions_and_labels.csv sample ===")
print(lab.head(3))

# 4) quick sanity: thử tìm keyword “leish” xem có data liên quan domain của bạn không
# (đừng kỳ vọng nhiều vì MultiCaRe là đa-chuyên-khoa, không riêng Leishmania)
print("\n=== quick search 'leish' in cases (may take a bit) ===")
try:
    q = f"""
    SELECT COUNT(*) AS n
    FROM read_parquet('{(root/'cases.parquet').as_posix()}')
    WHERE lower(CAST(* AS VARCHAR)) LIKE '%leish%'
    """
    # NOTE: query trên có thể fail tùy duckdb version/schema; nếu fail thì bạn search đúng cột text sau khi xem DESCRIBE.
    print(con.execute(q).df())
except Exception as e:
    print("Search failed (expected sometimes). After you see schema, search the main text column only. Error:", e)
