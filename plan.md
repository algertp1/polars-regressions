# Plan: fexp_panel2_parquet

## Goal
Create notebook `fexp_panel2_parquet.ipynb` that reads SAS v9 dataset
`Y:\sasdata\barra\gem4us\fexp_panel.sas7bdat`, keeps rows where
`aci_us = 1 OR aci_usmc = 1`, and writes filtered output to `./parquet_files/`.

## Why the earlier search hung
Initial repo-wide glob/grep scanned `.venv` recursively (large tree). Use targeted
listing or exclude `.venv` for this small project.

## Environment (uv)
- Project managed with `uv`; dependencies in `pyproject.toml`.
- Add `pyreadstat` for SAS v9 (`.sas7bdat`) reads.
- Existing: `polars`, `ipykernel`, `parquet`.

## Approach
1. **Load strategy**: SAS files do not support predicate pushdown like Parquet.
   Use manual `row_offset` / `row_limit` reads with
   `read_sas7bdat(..., output_format="polars")` (not `read_file_in_chunks`, which
   still probes metadata via pandas).
2. **Filter**: `(aci_us == 1) | (aci_usmc == 1)` on each chunk (Polars).
3. **Write**: Concat filtered chunks; `write_parquet()` to `./parquet_files/fexp_panel.parquet`.
4. **Paths**: Create `parquet_files/` if missing (already gitignored).

## Notebook cells
1. Markdown — purpose and inputs/outputs
2. Config — paths, chunk size
3. Chunked read → filter → write parquet
4. Quick validation — row count, schema, sample rows

## Status
- [x] Plan documented
- [x] Add `pyreadstat` via `uv add` (polars pinned to 1.42.0 for `exclude-newer`)
- [x] Create `fexp_panel2_parquet.ipynb`
- [x] Smoke-check imports (`uv run python -c "import pyreadstat, polars"`)
