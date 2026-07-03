# Plan: Barra_frets — Barra factor returns via lazy WLS

## Overview

Two-step pipeline: extract US/USMC names from SAS to Parquet, then run monthly
cross-sectional WLS factor regressions on the Parquet panel.

| Step | Notebook | Input | Output |
|------|----------|-------|--------|
| 1 | `fexp_panel2_parquet.ipynb` | SAS `fexp_panel.sas7bdat` | `./parquet_files/fexp_panel.parquet` |
| 2 | `Barra_frets.ipynb` | `./parquet_files/fexp_panel.parquet` | Monthly multivariate betas + fit stats |
| 3 | `Barra_frets.ipynb` | `./parquet_files/fexp_panel.parquet` | Monthly univariate (risk + industry) betas |

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

**Goal:** `Barra_frets.ipynb` (+ `Barra_frets.py` reference script) — lazy Polars
WLS of `ret` on Barra style + industry factors, one regression per `date` (month).

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

### Independent variables (62 total, no intercept)

**Style / risk factors (17):**

`WORLD`, `BETA`, `BTOP`, `DIVYILD`, `EARNQLTY`, `EARNVAR`, `EARNYILD`,
`GROWTH`, `INVSQLTY`, `LEVERAGE`, `LIQUIDTY`, `LTREVRSL`, `MIDCAP`,
`MOMENTUM`, `PROFIT`, `RESVOL`, `SIZE`

**Industry factors (45):**

`AEROSPCE`, `AIRLINES`, `DIVMETAL`, `AUTOCOMP`, `BANKS`, `BIOTECH`, `BLDCNSTR`,
`CHEMICAL`, `COMMSVCS`, `COMMUNIC`, `COMPUTER`, `CONSTPP`, `CONSDUR`, `CONSVCS`,
`DIVFIN`, `ENERGY`, `AGROCHEM`, `FOODPRD`, `FOODRETL`, `GOLD`, `HLTHEQP`,
`HLTHSVC`, `HSHLDPRD`, `INOILGAS`, `INSURNCE`, `INTERNET`, `SOFTWARE`,
`MACHINRY`, `MEDIA`, `OILGAS`, `OILEXPL`, `PHARMA`, `PRECMETL`, `REALEST`,
`RETAIL`, `SEMICOND`, `SMICNDEQ`, `STEEL`, `TELECOM`, `TRNSPORT`, `UTILITY`,
`CAPMRKTS`, `RGNLBNKS`, `THRIFTS`, `RLESTMNG`

**No intercept:** `WORLD` is 1 for all names and serves as the implicit intercept.

### Grouping

`group_by("date")` — one cross-section per month (~379 months in sample).

### WLS implementation

- `polars_ols` via `pl.col("ret").least_squares.wls(...)`.
- `add_intercept=False`, `null_policy="drop"`, `solve_method="svd"`.
- **Betas:** `mode="coefficients"` → struct unnested to one column per factor.
- **Summary stats:** optional (`INCLUDE_STATS=False` by default). When enabled,
  compute `sse`, `tss`, `r2` via a second WLS residuals pass (~2× regression time).
  `mode="statistics"` panics on this design matrix.

### Output shape

One row per `date` with columns:

- `date`, `n_obs`, `sse`, `tss`, `r2`
- 62 beta columns (factor names)

Optional: write results to `./parquet_files/fexp_wls_betas.parquet`.

### Notebook cells

1. Markdown — model spec
2. Config — paths, factor name lists
3. Build lazy query graph
4. Collect, unnest betas, preview
5. Optional save + sanity checks

### Status

- [x] Factor lists confirmed with user
- [x] Pipeline smoke-tested (379 months, 62 betas)
- [x] Create `Barra_frets.ipynb`
- [x] Align `Barra_frets.py` reference script

---

## Phase 3: Univariate risk-factor WLS (industry-controlled)

**Goal:** For each style/risk factor, run a separate monthly cross-sectional WLS with
industries as controls — one regression per `(date, risk_factor)`.

### Model (per risk factor)

`ret ~ risk_factor + industries` with `regwt = 1 / srisk²`, no intercept.

Example: `ret ~ MOMENTUM + AEROSPCE + … + RLESTMNG` grouped by `date`.

Industries are the same 45 `INDUSTRY_FACTORS` used in Phase 2. Each of the 17
`RISK_FACTORS` gets its own regression; only that factor's beta is retained per run.

### Output shape

One row per `date` with columns:

- `date`, `n_obs`
- 17 beta columns (one per risk factor, industry-controlled)

Write to `./parquet_files/fexp_wls_univariate_betas.parquet`.

### Downstream use

Recompute analytics from univariate betas (not the full multivariate model):

1. **Selected factor summary** — mean and Sharpe (`sqrt(12) × mean / stdev`) for
   `SUMMARY_FACTORS` subset.
2. **Trailing chart** — rolling `TRAIL_MONTHS` (36) monthly average of WLS betas for
   `PLOT_FACTORS` subset.

Both tables/charts subset columns from the univariate output; the full multivariate
betas in Phase 2 remain available separately.

### Performance

All 17 regressions share one lazy plan: a single parquet scan, filter, weight
column, and `group_by("date")`. Only the per-group WLS solves multiply.

Benchmark on this panel (~379 months):

| Approach | Wall time | vs multivariate |
|----------|-----------|-----------------|
| Phase 2 multivariate (1 WLS, 62 factors) | ~8 s | 1.0× |
| Phase 3 univariate (17 WLS, shared plan) | ~22 s | ~2.6× |
| Naive 17 separate `collect()` calls | ~102 s | ~12× |

So 17 regressions per month costs ~2.6× a full multivariate pass, not ~17×.
The lazy plan avoids re-reading parquet and re-grouping on each factor.

### Notebook cells

1. Markdown — univariate model spec
2. Build lazy query (17 WLS passes per month via `group_by("date")`)
3. Collect, preview, save parquet
4. Summary table (industry-controlled betas)
5. Trailing 3-year factor return chart

### Status

- [x] Factor lists aligned with Phase 2 (17 risk + 45 industry controls)
- [x] Notebook section + parquet output
- [x] Summary table and chart wired to univariate betas
