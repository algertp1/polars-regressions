"""Benchmark: polars + sklearn WLS vs pandas + sklearn WLS.

Two regimes are timed:

  MULTIVARIATE (Phase 2)
    ret ~ WORLD + BETA + ... + 45 industries  (62 factors, no intercept)
    One cross-section per month.

  UNIVARIATE (Phase 3)
    For each of 17 style factors: ret ~ style_factor + 45 industries  (46 features)
    17 separate regressions per month.

Each regime reports:
  - end-to-end time (parquet read + filter/prep + regression)
  - regression-only time (data pre-loaded; isolates sklearn loop cost)
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression

from barra_frets import (
    FACTOR_COLUMNS,
    INDUSTRY_FACTORS,
    PARQUET_PATH,
    RISK_FACTORS,
    load_regression_panel,
    run_multivariate_wls,
    run_univariate_wls,
)

RUNS = 3  # average over multiple runs to reduce jitter


def time_it(label: str, fn, runs: int = RUNS) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"  {label:<52} {best:6.2f}s  (best of {runs})")
    return best


def pandas_load_panel() -> pd.DataFrame:
    df = (
        pd.read_parquet(PARQUET_PATH)
        .query("country_gem4 == 'USA'")
        .dropna(subset=["ret", "srisk", *FACTOR_COLUMNS])
    )
    df["regwt"] = 1.0 / df["srisk"] ** 2
    return df


def pandas_multivariate_wls(df_pd: pd.DataFrame) -> dict:
    results = {}
    model = LinearRegression(fit_intercept=False)
    for date, grp in df_pd.groupby("date"):
        model.fit(
            grp[FACTOR_COLUMNS].values,
            grp["ret"].values,
            sample_weight=grp["regwt"].values,
        )
        results[date] = model.coef_
    return results


UNI_FEATURES: dict[str, list[str]] = {f: [f, *INDUSTRY_FACTORS] for f in RISK_FACTORS}
N_RISK = len(RISK_FACTORS)
N_IND_FEATURES = 1 + len(INDUSTRY_FACTORS)


def pandas_univariate_wls(df_pd: pd.DataFrame) -> dict:
    results: dict = {}
    model = LinearRegression(fit_intercept=False)
    for date, grp in df_pd.groupby("date"):
        y = grp["ret"].values
        w = grp["regwt"].values
        for factor, cols in UNI_FEATURES.items():
            model.fit(grp[cols].values, y, sample_weight=w)
            results[(date, factor)] = model.coef_[0]
    return results


# --------------------------------------------------------------------------- #
# Pre-load panels (exclude from regression-only timing)                       #
# --------------------------------------------------------------------------- #
print("Loading panels (not timed for regression-only runs)...")
df_pd = pandas_load_panel()
df_pl = load_regression_panel()
months = df_pd["date"].unique()
print(f"  {len(months):,} months, {len(df_pd):,} rows, {len(FACTOR_COLUMNS)} factors\n")

# --------------------------------------------------------------------------- #
# MULTIVARIATE                                                                #
# --------------------------------------------------------------------------- #
print("MULTIVARIATE — end-to-end (I/O + prep + regression)")
t_pl_multi_e2e = time_it(
    "polars scan/collect + sklearn map_groups",
    lambda: run_multivariate_wls(),
)
t_pd_multi_e2e = time_it(
    "pandas read_parquet + filter/prep + sklearn loop",
    lambda: pandas_multivariate_wls(pandas_load_panel()),
)

print("\nMULTIVARIATE — regression only (data pre-loaded)")
t_pl_multi = time_it(
    "polars panel + sklearn map_groups",
    lambda: run_multivariate_wls(panel=df_pl),
)
t_pd_multi = time_it(
    "pandas panel + sklearn loop",
    lambda: pandas_multivariate_wls(df_pd),
)

# --------------------------------------------------------------------------- #
# UNIVARIATE                                                                  #
# --------------------------------------------------------------------------- #
print(f"\n{'=' * 60}")
print(f"UNIVARIATE ({N_RISK} regressions/month × {N_IND_FEATURES} features each)")
print(f"{'=' * 60}")

print("\nUNIVARIATE — end-to-end (I/O + prep + regression)")
t_pl_uni_e2e = time_it(
    "polars scan/collect + sklearn univariate map_groups",
    lambda: run_univariate_wls(),
)
t_pd_uni_e2e = time_it(
    "pandas read_parquet + filter/prep + sklearn nested loop",
    lambda: pandas_univariate_wls(pandas_load_panel()),
)

print("\nUNIVARIATE — regression only (data pre-loaded)")
t_pl_uni = time_it(
    "polars panel + sklearn univariate map_groups",
    lambda: run_univariate_wls(panel=df_pl),
)
t_pd_uni = time_it(
    "pandas panel + sklearn nested loop",
    lambda: pandas_univariate_wls(df_pd),
)

# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #
baseline_multi = t_pl_multi_e2e
baseline_uni = t_pl_uni_e2e

print(f"""
{'=' * 60}
SUMMARY ({RUNS}-run best-of, {len(months):,} months)

MULTIVARIATE  (1 regression × {len(FACTOR_COLUMNS)} features)
  [end-to-end incl. I/O]
  polars + sklearn          {t_pl_multi_e2e:6.2f}s   1.0x  (baseline)
  pandas + sklearn          {t_pd_multi_e2e:6.2f}s  {t_pd_multi_e2e / baseline_multi:5.1f}x  vs polars

  [regression only — data pre-loaded]
  polars + sklearn          {t_pl_multi:6.2f}s   1.0x  (baseline)
  pandas + sklearn          {t_pd_multi:6.2f}s  {t_pd_multi / t_pl_multi:5.1f}x  vs polars

UNIVARIATE  ({N_RISK} regressions × {N_IND_FEATURES} features)
  [end-to-end incl. I/O]
  polars + sklearn          {t_pl_uni_e2e:6.2f}s   1.0x  (baseline)
  pandas + sklearn          {t_pd_uni_e2e:6.2f}s  {t_pd_uni_e2e / baseline_uni:5.1f}x  vs polars

  [regression only — data pre-loaded]
  polars + sklearn          {t_pl_uni:6.2f}s   1.0x  (baseline)
  pandas + sklearn          {t_pd_uni:6.2f}s  {t_pd_uni / t_pl_uni:5.1f}x  vs polars

SCALING: multivariate -> univariate ({N_RISK} regressions instead of 1)
  polars + sklearn (e2e)    {t_pl_uni_e2e / t_pl_multi_e2e:5.1f}x  (ideal: {N_RISK}x)
  pandas + sklearn (e2e)    {t_pd_uni_e2e / t_pd_multi_e2e:5.1f}x  (ideal: {N_RISK}x)
  polars + sklearn (loop)   {t_pl_uni / t_pl_multi:5.1f}x  (ideal: {N_RISK}x)
  pandas + sklearn (loop)   {t_pd_uni / t_pd_multi:5.1f}x  (ideal: {N_RISK}x)

Notes:
  - Both paths use sklearn LinearRegression(fit_intercept=False, sample_weight=...).
  - Polars keeps columnar I/O and group iteration; pandas materializes a full
    DataFrame before the sklearn loop.
  - Univariate scaling tracks ~{N_RISK}x because each month runs {N_RISK} separate
    sklearn fits in both implementations.
{'=' * 60}
""")
