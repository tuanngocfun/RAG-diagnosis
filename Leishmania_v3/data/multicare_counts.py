from pathlib import Path
import duckdb

root = Path("./whole_multicare_dataset")
cases_p = (root / "cases.parquet").as_posix()
imgs_p  = (root / "case_images.parquet").as_posix()
meta_p = (root / "metadata.parquet").as_posix()
abs_p = (root / "abstracts.parquet").as_posix()

con = duckdb.connect()
con.execute("INSTALL parquet; LOAD parquet;")

# Regex tuned for leishmaniasis + synonyms/species.
LEISH_PATTERN = (
    r"(leishman|leishmania|leishmaniasis|kala[- ]azar|post[- ]kala[- ]azar|pkdl|"
    r"visceral leish|cutaneous leish|mucocutaneous leish|"
    r"l\.?\s?(donovani|infantum|tropica|major|braziliensis|mexicana|panamensis|"
    r"guyanensis|amazonensis))"
)

q = f"""
WITH
  all_cases AS (
    SELECT
      rp.article_id::VARCHAR AS article_id,
      c.case_id::VARCHAR AS case_id,
      COALESCE(c.case_text::VARCHAR, '') AS case_text
    FROM read_parquet('{cases_p}') rp
    CROSS JOIN UNNEST(rp.cases) AS t(c)
  ),
  img_cases AS (
    SELECT
      ci.case_id::VARCHAR AS case_id,
      list_count(ci.case_image_list) AS n_images
    FROM read_parquet('{imgs_p}')
    CROSS JOIN UNNEST(case_images) AS t(ci)
  ),
  merged AS (
    SELECT
      a.case_id,
      a.case_text,
      COALESCE(i.n_images, 0) AS n_images
    FROM all_cases a
    LEFT JOIN img_cases i USING(case_id)
  )
SELECT
  COUNT(*) AS total_cases,
  SUM(CASE WHEN n_images > 0 THEN 1 ELSE 0 END) AS cases_text_plus_images,
  SUM(CASE WHEN n_images = 0 THEN 1 ELSE 0 END) AS cases_text_only,

  -- Leish presence in case_text (expandable regex)
  SUM(CASE WHEN regexp_matches(lower(case_text), '{LEISH_PATTERN}') THEN 1 ELSE 0 END) AS cases_mention_leishman,
  SUM(CASE WHEN regexp_matches(lower(case_text), '{LEISH_PATTERN}') AND n_images > 0 THEN 1 ELSE 0 END) AS leish_text_plus_images,
  SUM(CASE WHEN regexp_matches(lower(case_text), '{LEISH_PATTERN}') AND n_images = 0 THEN 1 ELSE 0 END) AS leish_text_only
FROM merged;
"""
print(con.execute(q).df())

# (tuỳ chọn) In ra vài case để kiểm tra đúng là leishmaniasis/leishmania
q2 = f"""
WITH
  all_cases AS (
    SELECT rp.article_id::VARCHAR AS article_id,
           c.case_id::VARCHAR AS case_id,
           COALESCE(c.case_text::VARCHAR, '') AS case_text
    FROM read_parquet('{cases_p}') rp
    CROSS JOIN UNNEST(rp.cases) AS t(c)
  ),
  img_cases AS (
    SELECT ci.case_id::VARCHAR AS case_id,
           list_count(ci.case_image_list) AS n_images
    FROM read_parquet('{imgs_p}')
    CROSS JOIN UNNEST(case_images) AS t(ci)
  )
SELECT
  a.case_id,
  i.n_images,
  substr(a.case_text, 1, 250) AS snippet
FROM all_cases a
LEFT JOIN img_cases i USING(case_id)
WHERE regexp_matches(lower(a.case_text), '{LEISH_PATTERN}')
LIMIT 10;
"""
print(con.execute(q2).df())

# (tuỳ chọn) Mở rộng: tìm theo abstract/metadata để không bỏ sót case_text im lặng
q3 = f"""
WITH
  all_cases AS (
    SELECT rp.article_id::VARCHAR AS article_id,
           c.case_id::VARCHAR AS case_id,
           COALESCE(c.case_text::VARCHAR, '') AS case_text
    FROM read_parquet('{cases_p}') rp
    CROSS JOIN UNNEST(rp.cases) AS t(c)
  ),
  img_cases AS (
    SELECT ci.case_id::VARCHAR AS case_id,
           list_count(ci.case_image_list) AS n_images
    FROM read_parquet('{imgs_p}')
    CROSS JOIN UNNEST(case_images) AS t(ci)
  ),
  meta_match AS (
    SELECT m.article_id::VARCHAR AS article_id
    FROM read_parquet('{meta_p}') m
    WHERE regexp_matches(
      lower(
        COALESCE(m.article_metadata.title::VARCHAR, '') || ' ' ||
        COALESCE(CAST(m.article_metadata.mesh_terms AS VARCHAR), '') || ' ' ||
        COALESCE(CAST(m.article_metadata.major_mesh_terms AS VARCHAR), '') || ' ' ||
        COALESCE(CAST(m.article_metadata.keywords AS VARCHAR), '')
      ),
      '{LEISH_PATTERN}'
    )
  ),
  abs_match AS (
    SELECT a.article_id::VARCHAR AS article_id
    FROM read_parquet('{abs_p}') a
    WHERE regexp_matches(lower(COALESCE(a.abstract::VARCHAR, '')), '{LEISH_PATTERN}')
  ),
  combined_articles AS (
    SELECT article_id FROM meta_match
    UNION
    SELECT article_id FROM abs_match
  ),
  matched_cases AS (
    SELECT ac.case_id, ac.article_id, ac.case_text
    FROM all_cases ac
    WHERE regexp_matches(lower(ac.case_text), '{LEISH_PATTERN}')
    UNION
    SELECT ac.case_id, ac.article_id, ac.case_text
    FROM all_cases ac
    JOIN combined_articles ca ON ac.article_id = ca.article_id
  )
SELECT
  COUNT(*) AS cases_from_case_text_or_article_context,
  SUM(CASE WHEN COALESCE(i.n_images, 0) > 0 THEN 1 ELSE 0 END) AS cases_with_images,
  SUM(CASE WHEN COALESCE(i.n_images, 0) = 0 THEN 1 ELSE 0 END) AS cases_text_only
FROM matched_cases mc
LEFT JOIN img_cases i USING(case_id);
"""
print(con.execute(q3).df())

# (xuất CSV) danh sách case_id + article_id + n_images để dùng cho RAG
q4 = f"""
WITH
  all_cases AS (
    SELECT rp.article_id::VARCHAR AS article_id,
           c.case_id::VARCHAR AS case_id,
           COALESCE(c.case_text::VARCHAR, '') AS case_text
    FROM read_parquet('{cases_p}') rp
    CROSS JOIN UNNEST(rp.cases) AS t(c)
  ),
  img_cases AS (
    SELECT ci.case_id::VARCHAR AS case_id,
           list_count(ci.case_image_list) AS n_images
    FROM read_parquet('{imgs_p}')
    CROSS JOIN UNNEST(case_images) AS t(ci)
  ),
  meta_match AS (
    SELECT m.article_id::VARCHAR AS article_id
    FROM read_parquet('{meta_p}') m
    WHERE regexp_matches(
      lower(
        COALESCE(m.article_metadata.title::VARCHAR, '') || ' ' ||
        COALESCE(CAST(m.article_metadata.mesh_terms AS VARCHAR), '') || ' ' ||
        COALESCE(CAST(m.article_metadata.major_mesh_terms AS VARCHAR), '') || ' ' ||
        COALESCE(CAST(m.article_metadata.keywords AS VARCHAR), '')
      ),
      '{LEISH_PATTERN}'
    )
  ),
  abs_match AS (
    SELECT a.article_id::VARCHAR AS article_id
    FROM read_parquet('{abs_p}') a
    WHERE regexp_matches(lower(COALESCE(a.abstract::VARCHAR, '')), '{LEISH_PATTERN}')
  ),
  combined_articles AS (
    SELECT article_id FROM meta_match
    UNION
    SELECT article_id FROM abs_match
  ),
  matched_cases AS (
    SELECT ac.case_id, ac.article_id
    FROM all_cases ac
    WHERE regexp_matches(lower(ac.case_text), '{LEISH_PATTERN}')
    UNION
    SELECT ac.case_id, ac.article_id
    FROM all_cases ac
    JOIN combined_articles ca ON ac.article_id = ca.article_id
  )
SELECT
  mc.case_id,
  mc.article_id,
  COALESCE(i.n_images, 0) AS n_images
FROM matched_cases mc
LEFT JOIN img_cases i USING(case_id)
ORDER BY mc.article_id, mc.case_id;
"""
out_csv = root / "leishmaniasis_matched_cases.csv"
df = con.execute(q4).df()
df.to_csv(out_csv, index=False)
print(f"Wrote CSV: {out_csv} (rows={len(df)})")
