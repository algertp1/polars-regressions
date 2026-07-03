# plfrets

Monthly Barra factor returns (frets) via lazy Polars and `polars-ols`.

This repo is a worked example of running large-scale cross-sectional regressions
efficiently: read a factor exposure panel once, build a lazy query graph, and let
Polars fuse I/O, filters, and grouping into a shared execution plan. The payoff
shows up when you need many related regressions on the same data — not 17× the
work for 17 regressions per month, but roughly 2–3×.

## Objective

Estimate Barra style and industry factor returns with weighted least squares
(WLS), one cross-section per month, on US / USMC names from the GEM4US exposure
panel.

The project demonstrates two ideas:

1. **Lazy evaluation** — build the full regression pipeline as a `LazyFrame`
   before calling `collect()`. Filters, weights, and aggregations stay in the
   query plan until execution, so Polars can optimize and push work down to the
   parquet scan.

2. **Shared execution plans** — multiple WLS fits in one `group_by("date").agg(...)`
   share a single parquet read, filter, weight column, and group boundary. Only
   the per-group linear algebra (SVD solves) scales with the number of models.

Phase 2 runs one multivariate WLS per month (`ret` on 17 style + 45 industry
factors). Phase 3 runs 17 industry-controlled univariate regressions per month
(`ret ~ risk_factor + industries`). On the sample panel (~378 months), Phase 2
takes ~8 seconds; Phase 3 takes ~26 seconds — not ~17× slower.

## Quick start

### 1. Environment

Requires Python 3.14+. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```powershell
uv sync
```

Select the project venv as the Jupyter kernel (`plfrets (.venv)`) before running
notebooks.

### 2. Build the parquet panel (first time, ~few minutes)

Open and run **`fexp_panel2_parquet.ipynb`**.

This reads the Barra SAS exposure file in chunks, keeps rows where
`aci_us = 1` or `aci_usmc = 1`, and writes:

```
parquet_files/fexp_panel.parquet
```

The default SAS path is `Y:\sasdata\barra\gem4us\fexp_panel.sas7bdat`. Edit
`SAS_PATH` in the notebook if your file lives elsewhere. This step is I/O-bound
and takes a few minutes; you only need to rerun it when the source data changes.

### 3. Run factor returns (~seconds)

Open and run **`barra_frets.ipynb`**.

With the parquet in place, monthly frets on the full Barra history complete in
a few seconds:

| Step | Model | Output |
|------|-------|--------|
| Phase 2 | `ret ~ all style + industry factors` | `parquet_files/fexp_wls_betas.parquet` |
| Phase 3 | `ret ~ one style factor + industries` (×17) | `parquet_files/fexp_wls_univariate_betas.parquet` |

The notebook also builds a factor mean/Sharpe summary and a trailing 3-year chart
from the industry-controlled univariate betas.

A CLI companion script runs the multivariate pass:

```powershell
uv run python barra_frets.py
```

See `plan.md` for model specification and factor lists.

## Why `polars-ols`

Regressions use [`polars-ols`](https://github.com/baseplate/polars-ols), which
registers `.least_squares` on Polars expressions. It covers the common cases in
one API:

- **OLS** — ordinary least squares
- **WLS** — weighted least squares (used here: `regwt = 1 / srisk²`)
- **Ridge** — L2-regularized least squares

All integrate with lazy Polars: expressions inside `group_by().agg()` participate
in the same optimized plan as the rest of the pipeline. That makes it practical
to swap weighting or add regularization without restructuring the workflow.

## Performance benchmark

`benchmark_sklearn.py` times three implementations against the same data
(378 months, 1.2 M rows, best-of-3 runs on Windows/Python 3.14).

### Multivariate — 1 regression × 62 features per month

| Implementation | Time | vs polars |
|----------------|-----:|----------:|
| polars-ols lazy | 8.15 s | 1.0× |
| pandas + numpy `lstsq` | 11.18 s | 1.4× |
| pandas + sklearn `LinearRegression` | 11.99 s | 1.5× |
| pandas full pipeline ¹ | 11.99 s | 1.5× |

### Univariate — 17 regressions × 46 features per month (Phase 3)

| Implementation | Time | vs polars | scaling from multivariate |
|----------------|-----:|----------:|--------------------------:|
| polars-ols lazy | 26.40 s | 1.0× | 3.2× |
| pandas + numpy nested | 111.63 s | 4.2× | 10.0× |
| pandas + sklearn nested | 122.55 s | 4.6× | 10.2× |
| pandas full pipeline ¹ | 112.34 s | 4.3× | — |

Polars scales sub-linearly (3.2×) because all 17 WLS expressions share
one lazy DAG — single parquet scan, single filter, single `group_by`.
Numpy and sklearn scale ~10× (not 17×) because multi-threaded LAPACK
absorbs some of the per-call cost, but Python's GIL serializes the 6,426
dispatch calls per run regardless.

> ¹ **"Full pipeline"** re-reads the parquet file inside the timed block.
> For polars the I/O is already fused into the lazy plan and therefore
> included in every timed run. The pandas approaches pre-load into a
> DataFrame and time only the regression loop, which understates
> end-to-end cost. The full-pipeline variant gives a like-for-like
> comparison and shows that at 17 regressions/month the I/O cost
> (< 1 s) is negligible relative to 6,426 LAPACK calls.

## Repository layout

| File | Purpose |
|------|---------|
| `fexp_panel2_parquet.ipynb` | SAS → filtered parquet (run first) |
| `barra_frets.ipynb` | Lazy WLS frets, summary, charts |
| `barra_frets.py` | Reference script for multivariate WLS |
| `benchmark_sklearn.py` | polars vs numpy vs sklearn timing (multivariate + univariate) |
| `plan.md` | Detailed model spec and pipeline notes |
| `parquet_files/` | Generated parquet outputs (gitignored) |

## Notes

- Phase 2 uses no intercept; `WORLD` is 1 for all names and acts as the implicit
  intercept.
- Phase 3 controls for industries in each univariate regression; summary and
  chart cells subset to selected factors via `SUMMARY_FACTORS` and `PLOT_FACTORS`.
- Optional fit statistics (`sse`, `tss`, `r2`) in Phase 2 require a second WLS
  pass per month and roughly double runtime — off by default.
