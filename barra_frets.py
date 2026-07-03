"""Barra_frets — monthly Barra factor WLS via lazy Polars.

Reference script companion to ``Barra_frets.ipynb``. Reads
``parquet_files/fexp_panel.parquet`` and runs cross-sectional WLS by date.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import polars_ols  # noqa: F401 — registers .least_squares namespace

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

PLOT_FACTORS = ["MOMENTUM", "BETA", "RESVOL", "SIZE", "EARNYILD"]

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


def wls_expr(features: list[str], mode: str, *, add_intercept: bool = False) -> pl.Expr:
    return pl.col("ret").least_squares.wls(
        *[pl.col(name) for name in features],
        sample_weights=pl.col("regwt"),
        add_intercept=add_intercept,
        null_policy="drop",
        solve_method="svd",
        mode=mode,
    )


def build_lazy_wls_plan(
    parquet_path: Path = PARQUET_PATH,
    *,
    include_stats: bool = INCLUDE_STATS,
) -> pl.LazyFrame:
    """Build the Barra_frets lazy WLS query (one regression per ``date``)."""
    weighted_mean_ret = (pl.col("ret") * pl.col("regwt")).sum() / pl.col("regwt").sum()

    agg_exprs: list[pl.Expr] = [
        pl.len().alias("n_obs"),
        wls_expr(FACTOR_COLUMNS, "coefficients").alias("betas"),
    ]
    if include_stats:
        agg_exprs.extend(
            [
                (pl.col("regwt") * wls_expr(FACTOR_COLUMNS, "residuals").pow(2))
                .sum()
                .alias("sse"),
                (pl.col("regwt") * (pl.col("ret") - weighted_mean_ret).pow(2))
                .sum()
                .alias("tss"),
            ]
        )

    plan = (
        _scan_regression_panel(parquet_path)
        .group_by("date")
        .agg(*agg_exprs)
        .sort("date")
    )
    if include_stats:
        plan = plan.with_columns((1.0 - pl.col("sse") / pl.col("tss")).alias("r2"))
    return plan


def _scan_regression_panel(parquet_path: Path = PARQUET_PATH) -> pl.LazyFrame:
    return (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("country_gem4") == "USA")
        .with_columns((1.0 / pl.col("srisk").pow(2)).alias("regwt"))
    )


def univariate_risk_beta_expr(risk_factor: str) -> pl.Expr:
    """WLS beta for one risk factor with industry controls."""
    features = [risk_factor, *INDUSTRY_FACTORS]
    return (
        wls_expr(features, "coefficients")
        .struct.field(risk_factor)
        .alias(risk_factor)
    )


def build_lazy_univariate_wls_plan(
    parquet_path: Path = PARQUET_PATH,
) -> pl.LazyFrame:
    """Build lazy WLS plan: one industry-controlled regression per risk factor per date."""
    return (
        _scan_regression_panel(parquet_path)
        .group_by("date")
        .agg(
            pl.len().alias("n_obs"),
            *[univariate_risk_beta_expr(factor) for factor in RISK_FACTORS],
        )
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
    results = build_lazy_wls_plan(include_stats=include_stats).collect()
    flat = results.unnest("betas")
    flat.write_parquet(OUTPUT_PATH)
    return flat


def main_univariate() -> pl.DataFrame:
    """Run industry-controlled univariate WLS and write betas to ``UNIVARIATE_OUTPUT_PATH``."""
    univariate = build_lazy_univariate_wls_plan().collect()
    univariate.write_parquet(UNIVARIATE_OUTPUT_PATH)
    return univariate


if __name__ == "__main__":
    df = main()
    print(f"Wrote {df.height:,} monthly regressions to {OUTPUT_PATH.resolve()}")
