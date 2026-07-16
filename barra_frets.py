"""Barra_frets — monthly Barra factor WLS via Polars I/O and sklearn WLS.

Reference script companion to ``barra_frets.ipynb``. Reads
``parquet_files/fexp_panel.parquet``, prepares the panel with lazy Polars,
and runs cross-sectional weighted least squares by ``date`` using scikit-learn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

PARQUET_PATH = Path("parquet_files/fexp_panel.parquet")
OUTPUT_PATH = Path("parquet_files/fexp_wls_betas.parquet")
UNIVARIATE_OUTPUT_PATH = Path("parquet_files/fexp_wls_univariate_betas.parquet")

INCLUDE_STATS = False
TRAIL_MONTHS = 36  # trailing 3-year window

RISK_FACTORS = [
    "WORLD",
    "BETA",
    "BTOP",
    "DIVYILD",
    "EARNQLTY",
    "EARNVAR",
    "EARNYILD",
    "GROWTH",
    "INVSQLTY",
    "LEVERAGE",
    "LIQUIDTY",
    "LTREVRSL",
    "MIDCAP",
    "MOMENTUM",
    "PROFIT",
    "RESVOL",
    "SIZE",
]

INDUSTRY_FACTORS = [
    "AEROSPCE",
    "AIRLINES",
    "DIVMETAL",
    "AUTOCOMP",
    "BANKS",
    "BIOTECH",
    "BLDCNSTR",
    "CHEMICAL",
    "COMMSVCS",
    "COMMUNIC",
    "COMPUTER",
    "CONSTPP",
    "CONSDUR",
    "CONSVCS",
    "DIVFIN",
    "ENERGY",
    "AGROCHEM",
    "FOODPRD",
    "FOODRETL",
    "GOLD",
    "HLTHEQP",
    "HLTHSVC",
    "HSHLDPRD",
    "INOILGAS",
    "INSURNCE",
    "INTERNET",
    "SOFTWARE",
    "MACHINRY",
    "MEDIA",
    "OILGAS",
    "OILEXPL",
    "PHARMA",
    "PRECMETL",
    "REALEST",
    "RETAIL",
    "SEMICOND",
    "SMICNDEQ",
    "STEEL",
    "TELECOM",
    "TRNSPORT",
    "UTILITY",
    "CAPMRKTS",
    "RGNLBNKS",
    "THRIFTS",
    "RLESTMNG",
]

PLOT_FACTORS = ["MOMENTUM", "BETA", "EARNYILD"]

SUMMARY_FACTORS = [
    "BETA",
    "WORLD",
    "GROWTH",
    "EARNYILD",
    "MOMENTUM",
    "SIZE",
    "PROFIT",
    "RESVOL",
]

FACTOR_COLUMNS = RISK_FACTORS + INDUSTRY_FACTORS


def wls_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    model: LinearRegression | None = None,
) -> np.ndarray:
    """Fit weighted least squares (no intercept) and return coefficients."""
    reg = model if model is not None else LinearRegression(fit_intercept=False)
    reg.fit(X, y, sample_weight=sample_weight)
    return reg.coef_


def _scan_regression_panel(parquet_path: Path = PARQUET_PATH) -> pl.LazyFrame:
    return (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("country_gem4") == "USA")
        .with_columns((1.0 / pl.col("srisk").pow(2)).alias("regwt"))
    )


def load_regression_panel(parquet_path: Path = PARQUET_PATH) -> pl.DataFrame:
    """Load filtered exposure panel with regression weights (lazy scan + collect)."""
    subset = ["ret", "srisk", *FACTOR_COLUMNS]
    return _scan_regression_panel(parquet_path).drop_nulls(subset=subset).collect()


def _fit_multivariate_group(
    group: pl.DataFrame,
    *,
    include_stats: bool = False,
) -> pl.DataFrame:
    subset = ["ret", "regwt", *FACTOR_COLUMNS]
    clean = group.drop_nulls(subset=subset)
    date = group["date"][0]
    n_obs = clean.height
    X = clean.select(FACTOR_COLUMNS).to_numpy()
    y = clean["ret"].to_numpy()
    w = clean["regwt"].to_numpy()
    model = LinearRegression(fit_intercept=False)
    coef = wls_coefficients(X, y, w, model=model)
    row: dict = {"date": date, "n_obs": n_obs}
    row.update(dict(zip(FACTOR_COLUMNS, coef, strict=True)))
    if include_stats:
        resid = y - X @ coef
        sse = float(np.sum(w * resid**2))
        wmean = float(np.sum(w * y) / np.sum(w))
        tss = float(np.sum(w * (y - wmean) ** 2))
        row.update(sse=sse, tss=tss, r2=(1.0 - sse / tss if tss else float("nan")))
    return pl.DataFrame([row])


def run_multivariate_wls(
    parquet_path: Path = PARQUET_PATH,
    *,
    include_stats: bool = INCLUDE_STATS,
    panel: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run multivariate monthly WLS (one regression per ``date``)."""
    data = panel if panel is not None else load_regression_panel(parquet_path)
    return (
        data.group_by("date", maintain_order=True)
        .map_groups(lambda g: _fit_multivariate_group(g, include_stats=include_stats))
        .sort("date")
    )


def _fit_univariate_group(group: pl.DataFrame) -> pl.DataFrame:
    subset = ["ret", "regwt", *FACTOR_COLUMNS]
    clean = group.drop_nulls(subset=subset)
    date = group["date"][0]
    n_obs = clean.height
    y = clean["ret"].to_numpy()
    w = clean["regwt"].to_numpy()
    model = LinearRegression(fit_intercept=False)
    row: dict = {"date": date, "n_obs": n_obs}
    for factor in RISK_FACTORS:
        cols = [factor, *INDUSTRY_FACTORS]
        X = clean.select(cols).to_numpy()
        model.fit(X, y, sample_weight=w)
        row[factor] = model.coef_[0]
    return pl.DataFrame([row])


def run_univariate_wls(
    parquet_path: Path = PARQUET_PATH,
    *,
    panel: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run industry-controlled univariate WLS (17 regressions per ``date``)."""
    data = panel if panel is not None else load_regression_panel(parquet_path)
    return (
        data.group_by("date", maintain_order=True)
        .map_groups(_fit_univariate_group)
        .sort("date")
    )


def factor_mean_sharpe_summary(
    monthly_betas: pl.DataFrame,
    factors: list[str] = SUMMARY_FACTORS,
) -> pl.DataFrame:
    """Mean and annualized Sharpe of monthly factor betas."""
    return (
        monthly_betas.select(factors)
        .unpivot(on=factors, variable_name="factor", value_name="beta")
        .group_by("factor")
        .agg(
            pl.col("beta").mean().alias("mean"),
            (pl.col("beta").mean() / pl.col("beta").std() * (12**0.5)).alias("sharpe"),
        )
        .with_columns(
            pl.col("mean").round(6),
            pl.col("sharpe").round(3),
        )
        .sort(
            pl.col("factor").replace_strict(
                {name: idx for idx, name in enumerate(factors)},
                default=len(factors),
            )
        )
    )


def plot_factor_trailing_returns(
    monthly_betas: pl.DataFrame,
    *,
    factors: list[str] = PLOT_FACTORS,
    trail_months: int = TRAIL_MONTHS,
) -> pl.DataFrame:
    """Trailing monthly average of WLS betas."""
    return monthly_betas.sort("date").select(
        "date",
        *[
            pl.col(f)
            .rolling_mean(window_size=trail_months, min_samples=trail_months)
            .alias(f)
            for f in factors
        ],
    )


def main(*, include_stats: bool = INCLUDE_STATS) -> pl.DataFrame:
    """Run Barra_frets WLS and write monthly betas to ``OUTPUT_PATH``."""
    flat = run_multivariate_wls(include_stats=include_stats)
    flat.write_parquet(OUTPUT_PATH)
    return flat


def main_univariate() -> pl.DataFrame:
    """Run industry-controlled univariate WLS and write betas to ``UNIVARIATE_OUTPUT_PATH``."""
    univariate = run_univariate_wls()
    univariate.write_parquet(UNIVARIATE_OUTPUT_PATH)
    return univariate


if __name__ == "__main__":
    df = main()
    print(f"Wrote {df.height:,} monthly regressions to {OUTPUT_PATH.resolve()}")
