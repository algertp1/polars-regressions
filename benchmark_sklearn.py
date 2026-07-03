"""Benchmark: polars-ols lazy WLS vs pandas + sklearn WLS.

Two regimes are timed:

  MULTIVARIATE (Phase 2)
    ret ~ WORLD + BETA + ... + 45 industries  (62 factors, no intercept)
    One cross-section per month.

  UNIVARIATE (Phase 3)
    For each of 17 style factors: ret ~ style_factor + 45 industries  (46 features)
    17 separate regressions per month, polars shares one lazy plan / single scan.

Three implementations per regime:
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
    INDUSTRY_FACTORS,
    PARQUET_PATH,
    RISK_FACTORS,
    build_lazy_univariate_wls_plan,
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
MULTIVARIATE results ({RUNS}-run best-of, {len(months):,} months, {len(FACTOR_COLUMNS)} factors):

  [regression loop only — data pre-loaded]
  polars-ols lazy           {t_polars:6.2f}s   1.0x  (baseline)
  pandas + sklearn          {t_sklearn:6.2f}s  {t_sklearn/t_polars:5.1f}x  slower
  pandas + numpy lstsq      {t_numpy:6.2f}s  {t_numpy/t_polars:5.1f}x  slower

  [end-to-end: I/O + filter + data prep + regression]
  polars lazy (same run)    {t_polars:6.2f}s   1.0x  (polars I/O included)
  pandas full pipeline      {t_e2e:6.2f}s  {t_e2e/t_polars:5.1f}x  slower
{'='*60}
""")

# --------------------------------------------------------------------------- #
# UNIVARIATE SECTION — 17 industry-controlled regressions per month           #
# --------------------------------------------------------------------------- #
# Each regression: ret ~ risk_factor + 45 industries  (46 features, no intercept)
# Only the target risk-factor coefficient is retained per regression.
# Polars: one lazy plan, single parquet scan, 17 WLS exprs in one group_by.agg()
# Numpy/sklearn: nested loop — (n_months × 17) separate lstsq / model.fit calls.

UNI_FEATURES: dict[str, list[str]] = {f: [f, *INDUSTRY_FACTORS] for f in RISK_FACTORS}
N_RISK = len(RISK_FACTORS)
N_IND_FEATURES = 1 + len(INDUSTRY_FACTORS)  # risk factor + 45 industries

print(f"\n{'='*60}")
print(f"UNIVARIATE benchmark ({N_RISK} regressions/month × {N_IND_FEATURES} features each)")
print(f"{'='*60}")

# ── Approach 5: polars-ols univariate lazy ─────────────────────────────────
print("\nApproach 5: polars-ols univariate lazy plan (single scan, 17 WLS exprs)")
t_polars_uni = time_it(
    "polars univariate collect()",
    lambda: build_lazy_univariate_wls_plan().collect(),
)

# ── Approach 6: pandas groupby + sklearn (nested month × factor loop) ──────
print("\nApproach 6: pandas groupby + sklearn (nested month × factor loop)")

def sklearn_univariate_loop() -> dict:
    results: dict = {}
    model = LinearRegression(fit_intercept=False)
    for date, grp in df_pd.groupby("date"):
        y = grp["ret"].values
        w = grp["regwt"].values
        for factor, cols in UNI_FEATURES.items():
            X = grp[cols].values
            model.fit(X, y, sample_weight=w)
            results[(date, factor)] = model.coef_[0]
    return results

t_sklearn_uni = time_it("pandas groupby + sklearn nested loop", sklearn_univariate_loop)

# ── Approach 7: pandas groupby + numpy lstsq (nested month × factor loop) ──
print("\nApproach 7: pandas groupby + numpy WLS (nested month × factor loop)")

def numpy_univariate_loop() -> dict:
    results: dict = {}
    for date, grp in df_pd.groupby("date"):
        y = grp["ret"].values
        sw = np.sqrt(grp["regwt"].values)
        yw = y * sw
        for factor, cols in UNI_FEATURES.items():
            X = grp[cols].values
            Xw = X * sw[:, None]
            coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            results[(date, factor)] = coef[0]
    return results

t_numpy_uni = time_it("pandas groupby + numpy lstsq nested loop", numpy_univariate_loop)

# ── Approach 8: pandas full pipeline univariate (I/O included) ────────────
print("\nApproach 8: pandas full pipeline univariate (read parquet + prep + nested loop)")

UNI_NEEDED_COLS = ["date", "country_gem4", "ret", "srisk", *FACTOR_COLUMNS]

def pandas_univariate_end_to_end() -> dict:
    df = (
        pd.read_parquet(PARQUET_PATH, columns=UNI_NEEDED_COLS)
        .query("country_gem4 == 'USA'")
        .dropna(subset=["ret", "srisk", *FACTOR_COLUMNS])
    )
    df["regwt"] = 1.0 / df["srisk"] ** 2
    results: dict = {}
    for date, grp in df.groupby("date"):
        y = grp["ret"].values
        sw = np.sqrt(grp["regwt"].values)
        yw = y * sw
        for factor, cols in UNI_FEATURES.items():
            X = grp[cols].values
            Xw = X * sw[:, None]
            coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            results[(date, factor)] = coef[0]
    return results

t_e2e_uni = time_it(
    "pandas full univariate pipeline (I/O + prep + nested loop)",
    pandas_univariate_end_to_end,
)

# --------------------------------------------------------------------------- #
# Combined summary                                                             #
# --------------------------------------------------------------------------- #
print(f"""
{'='*60}
SUMMARY ({RUNS}-run best-of)

MULTIVARIATE  ({len(months):,} months × 1 regression × {len(FACTOR_COLUMNS)} features)
  [regression only — data pre-loaded]
  polars-ols lazy           {t_polars:6.2f}s   1.0x  (baseline)
  pandas + sklearn          {t_sklearn:6.2f}s  {t_sklearn/t_polars:5.1f}x  slower
  pandas + numpy lstsq      {t_numpy:6.2f}s  {t_numpy/t_polars:5.1f}x  slower
  [end-to-end incl. I/O]
  pandas full pipeline      {t_e2e:6.2f}s  {t_e2e/t_polars:5.1f}x  vs polars lazy

UNIVARIATE  ({len(months):,} months × {N_RISK} regressions × {N_IND_FEATURES} features)
  [regression only — data pre-loaded]
  polars-ols univariate     {t_polars_uni:6.2f}s   1.0x  (baseline)
  pandas + sklearn nested   {t_sklearn_uni:6.2f}s  {t_sklearn_uni/t_polars_uni:5.1f}x  slower
  pandas + numpy nested     {t_numpy_uni:6.2f}s  {t_numpy_uni/t_polars_uni:5.1f}x  slower
  [end-to-end incl. I/O]
  pandas full pipeline      {t_e2e_uni:6.2f}s  {t_e2e_uni/t_polars_uni:5.1f}x  vs polars lazy

SCALING: multivariate -> univariate ({N_RISK} regressions instead of 1)
  polars-ols                {t_polars_uni/t_polars:5.1f}x  (ideal: {N_RISK}x,  actual: {t_polars_uni/t_polars:.1f}x)
  pandas + sklearn          {t_sklearn_uni/t_sklearn:5.1f}x  (ideal: {N_RISK}x,  actual: {t_sklearn_uni/t_sklearn:.1f}x)
  pandas + numpy            {t_numpy_uni/t_numpy:5.1f}x  (ideal: {N_RISK}x,  actual: {t_numpy_uni/t_numpy:.1f}x)

Notes:
  - Polars schedules all 17 WLS expressions in one lazy DAG; the parquet
    scan, filter, and group_by are executed once — hence sub-linear scaling.
  - Numpy/sklearn issue one LAPACK SVD call per (month × factor) pair.
    Scaling should track ~{N_RISK}x unless BLAS thread pools absorb the load.
  - Polars advantage compounds: I/O fusion + parallel expr evaluation vs
    Python loop overhead × 17.
{'='*60}
""")
