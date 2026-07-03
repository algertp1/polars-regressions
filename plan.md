# Plan: plfrets — Barra factor returns via lazy WLS

## Overview

Two-step pipeline: extract US/USMC names from SAS to Parquet, then run monthly
cross-sectional WLS factor regressions on the Parquet panel.

| Step | Notebook | Input | Output |
|------|----------|-------|--------|
| 1 | `fexp_panel2_parquet.ipynb` | SAS `fexp_panel.sas7bdat` | `./parquet_files/fexp_panel.parquet` |
| 2 | `plfrets.ipynb` | `./parquet_files/fexp_panel.parquet` | Monthly betas + fit stats |

## Environment (uv)

- Managed with `uv`; deps in `pyproject.toml`.
- Core: `polars`, `polars-ols`, `ipykernel`, `pyreadstat`.

---

## Phase 1: SAS → Parquet ✅

**Goal:** Keep rows where `aci_us = 1 OR aci_usmc = 1`.

**Approach:**
- Chunked SAS read via `read_sas7bdat(..., output_format="polars")`.
- Avoid `read_file_in_chunks` (metadata probe requires pandas).
- Write `./parquet_files/fexp_panel.parquet`.

**Status:** Done — see `fexp_panel2_parquet.ipynb`.

---

## Phase 2: Monthly WLS factor regressions

**Goal:** Turn `plfrets.py` into `plfrets.ipynb` — lazy Polars WLS of `ret` on
Barra style + industry factors, one regression per `date` (month).

### Input

`./parquet_files/fexp_panel.parquet` (~1.2M rows after Phase 1 filter; all
`country_gem4 = "USA"` in current extract).

### Row filters (lazy, predicate pushdown)

1. `country_gem4 == "USA"` — placeholder for pushdown filters (no-op on current file).
2. Future filters can be added here without changing downstream logic.

### Regression weight

`regwt = 1 / srisk^2`

### Dependent variable

`ret`

### Independent variables (56 total, no intercept)

**Style / risk factors (11):**

`WORLD`, `BETA`, `BTOP`, `DIVYILD`, `EARNQLTY`, `EARNVAR`, `EARNYILD`,
`GROWTH`, `INVSQLTY`, `LEVERAGE`, `LIQUIDTY`

**Industry factors (45):**

`AEROSPCE`, `AIRLINES`, `DIVMETAL`, `AUTOCOMP`, `BANKS`, `BIOTECH`, `BLDCNSTR`,
`CHEMICAL`, `COMMSVCS`, `COMMUNIC`, `COMPUTER`, `CONSTPP`, `CONSDUR`, `CONSVCS`,
`DIVFIN`, `ENERGY`, `AGROCHEM`, `FOODPRD`, `FOODRETL`, `GOLD`, `HLTHEQP`,
`HLTHSVC`, `HSHLDPRD`, `INOILGAS`, `INSURNCE`, `INTERNET`, `SOFTWARE`,
`MACHINRY`, `MEDIA`, `OILGAS`, `OILEXPL`, `PHARMA`, `PRECMETL`, `REALEST`,
`RETAIL`, `SEMICOND`, `SMICNDEQ`, `STEEL`, `TELECOM`, `TRNSPORT`, `UTILITY`,
`CAPMRKTS`, `RGNLBNKS`, `THRIFTS`, `RLESTMNG`

**No intercept:** `WORLD` is 1 for all names and serves as the implicit intercept.

**Excluded style columns** present in Parquet but not in this model:
`LTREVRSL`, `MIDCAP`, `MOMENTUM`, `PROFIT`, `RESVOL`, `SIZE`.

### Grouping

`group_by("date")` — one cross-section per month (~379 months in sample).

### WLS implementation

- `polars_ols` via `pl.col("ret").least_squares.wls(...)`.
- `add_intercept=False`, `null_policy="drop"`, `solve_method="svd"`.
- **Betas:** `mode="coefficients"` → struct unnested to one column per factor.
- **Summary stats:** `mode="statistics"` panics on this rank-deficient Barra
  design matrix; instead compute per month:
  - `n_obs` — row count
  - `sse` — Σ regwt × residual² (`mode="residuals"`)
  - `tss` — Σ regwt × (ret − weighted_mean(ret))²
  - `r2` — 1 − sse/tss

### Output shape

One row per `date` with columns:

- `date`, `n_obs`, `sse`, `tss`, `r2`
- 56 beta columns (factor names)

Optional: write results to `./parquet_files/fexp_wls_betas.parquet`.

### Notebook cells

1. Markdown — model spec
2. Config — paths, factor name lists
3. Build lazy query graph
4. Collect, unnest betas, preview
5. Optional save + sanity checks

### Status

- [x] Factor lists confirmed with user
- [x] Pipeline smoke-tested (379 months, 56 betas)
- [x] Create `plfrets.ipynb`
- [x] Align `plfrets.py` reference script
