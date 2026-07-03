"""Benchmark: polars-ols lazy WLS vs pandas + sklearn WLS (multivariate).

Runs the same model:
  ret ~ WORLD + BETA + ... + 45 industries  (62 factors, no intercept)
weighted by 1/srisk^2, one cross-section per month.

Three approaches timed:
  1. polars-ols  -- lazy plan, one collect()
  2. pandas loop -- groupby month, sklearn LinearRegression(fit_intercept=False)
  3. pandas loop -- groupby month, numpy lstsq with weights (faster than sklearn)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import polars_ols  # noqa: F401
from sklearn.linear_model import LinearRegression

from barra_frets import (
    FACTOR_COLUMNS,
    PARQUET_PATH,
    build_lazy_wls_plan,
)

RUNS = 3  # average over multiple runs to reduce jitter


def time_it(label: str, fn, runs: int = RUNS) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"  {label:<45} {best:6.2f}s  (best of {runs})")
    return best


# --------------------------------------------------------------------------- #
# Load data once into pandas (exclude from timing)                            #
# --------------------------------------------------------------------------- #
print("Loading parquet into pandas (not timed)...")
df_pd = (
    pd.read_parquet(PARQUET_PATH)
    .query("country_gem4 == 'USA'")
    .dropna(subset=["ret", "srisk", *FACTOR_COLUMNS])
)
df_pd["regwt"] = 1.0 / df_pd["srisk"] ** 2
months = df_pd["date"].unique()
print(f"  {len(months):,} months, {len(df_pd):,} rows, {len(FACTOR_COLUMNS)} factors\n")

# --------------------------------------------------------------------------- #
# Approach 1 — polars-ols lazy                                                #
# --------------------------------------------------------------------------- #
print("Approach 1: polars-ols lazy plan")
t_polars = time_it(
    "polars lazy collect()",
    lambda: build_lazy_wls_plan().collect().unnest("betas"),
)

# --------------------------------------------------------------------------- #
# Approach 2 — pandas groupby + sklearn WLS                                   #
# --------------------------------------------------------------------------- #
print("\nApproach 2: pandas groupby + sklearn LinearRegression")

def sklearn_loop():
    results = {}
    model = LinearRegression(fit_intercept=False)
    for date, grp in df_pd.groupby("date"):
        X = grp[FACTOR_COLUMNS].values
        y = grp["ret"].values
        w = grp["regwt"].values
        model.fit(X, y, sample_weight=w)
        results[date] = model.coef_
    return results

t_sklearn = time_it("pandas groupby + sklearn fit (loop)", sklearn_loop)

# --------------------------------------------------------------------------- #
# Approach 3 — pandas groupby + numpy lstsq (manual WLS)                     #
# --------------------------------------------------------------------------- #
print("\nApproach 3: pandas groupby + numpy WLS (lstsq on sqrt-weighted matrix)")

def numpy_loop():
    results = {}
    for date, grp in df_pd.groupby("date"):
        X = grp[FACTOR_COLUMNS].values
        y = grp["ret"].values
        sw = np.sqrt(grp["regwt"].values)
        Xw = X * sw[:, None]
        yw = y * sw
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        results[date] = coef
    return results

t_numpy = time_it("pandas groupby + numpy lstsq (loop)", numpy_loop)

# --------------------------------------------------------------------------- #
# Approach 4 — pandas including its own I/O + prep (end-to-end fair test)    #
# --------------------------------------------------------------------------- #
print("\nApproach 4: pandas full pipeline (read parquet + filter + prep + loop)")

def pandas_end_to_end():
    df = (
        pd.read_parquet(PARQUET_PATH, columns=["date", "country_gem4", "ret", "srisk", *FACTOR_COLUMNS])
        .query("country_gem4 == 'USA'")
        .dropna(subset=["ret", "srisk", *FACTOR_COLUMNS])
    )
    df["regwt"] = 1.0 / df["srisk"] ** 2
    results = {}
    for date, grp in df.groupby("date"):
        X = grp[FACTOR_COLUMNS].values
        y = grp["ret"].values
        sw = np.sqrt(grp["regwt"].values)
        coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        results[date] = coef
    return results

t_e2e = time_it("pandas full pipeline (I/O + filter + prep + loop)", pandas_end_to_end)

# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #
print(f"""
{'='*60}
Results ({RUNS}-run best-of, {len(months):,} months, {len(FACTOR_COLUMNS)} factors):

  [regression loop only — data pre-loaded]
  polars-ols lazy           {t_polars:6.2f}s   1.0x  (baseline)
  pandas + sklearn          {t_sklearn:6.2f}s  {t_sklearn/t_polars:5.1f}x  slower
  pandas + numpy lstsq      {t_numpy:6.2f}s  {t_numpy/t_polars:5.1f}x  slower

  [end-to-end: I/O + filter + data prep + regression]
  polars lazy (same run)    {t_polars:6.2f}s   1.0x  (polars I/O included)
  pandas full pipeline      {t_e2e:6.2f}s  {t_e2e/t_polars:5.1f}x  slower

Notes:
  - Both use LAPACK (numpy/scipy) for the actual SVD solves.
  - Polars advantage grows with I/O and data wrangling, which is fused
    into the lazy plan (one parquet scan, one filter, one groupby).
  - Pandas re-reads, re-filters, and re-groups on every full-pipeline call.
  - For 17 univariate models/month, polars shares the same single scan;
    pandas would need 17 separate loops or a more complex design.
{'='*60}
""")
