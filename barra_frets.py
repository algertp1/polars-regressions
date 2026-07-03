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
        pl.scan_parquet(parquet_path)
        .filter(pl.col("country_gem4") == "USA")
        .with_columns((1.0 / pl.col("srisk").pow(2)).alias("regwt"))
        .group_by("date")
        .agg(*agg_exprs)
        .sort("date")
    )
    if include_stats:
        plan = plan.with_columns((1.0 - pl.col("sse") / pl.col("tss")).alias("r2"))
    return plan


def build_extra_plot_factor_betas(
    factors: list[str],
    parquet_path: Path = PARQUET_PATH,
) -> pl.LazyFrame:
    """Single-factor WLS per month for style factors outside the main model."""
    agg_exprs = [
        wls_expr([factor], "coefficients", add_intercept=True)
        .struct.field(factor)
        .alias(factor)
        for factor in factors
    ]
    return (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("country_gem4") == "USA")
        .with_columns((1.0 / pl.col("srisk").pow(2)).alias("regwt"))
        .group_by("date")
        .agg(*agg_exprs)
        .sort("date")
    )


def risk_factor_summary(monthly_betas: pl.DataFrame) -> pl.DataFrame:
    """Mean and annualized Sharpe (sqrt(12) * mean / stdev) for style-factor betas."""
    return (
        monthly_betas.select(RISK_FACTORS)
        .unpivot(on=RISK_FACTORS, variable_name="factor", value_name="beta")
        .group_by("factor")
        .agg(
            pl.col("beta").mean().alias("mean"),
            (pl.col("beta").mean() / pl.col("beta").std() * (12**0.5)).alias("sharpe"),
        )
        .sort("factor")
    )


def plot_factor_trailing_returns(
    monthly_betas: pl.DataFrame,
    *,
    parquet_path: Path = PARQUET_PATH,
    trail_months: int = TRAIL_MONTHS,
) -> pl.DataFrame:
    """Merge main-model and supplemental betas; compute trailing monthly average."""
    from_model = [f for f in PLOT_FACTORS if f in monthly_betas.columns]
    extra = [f for f in PLOT_FACTORS if f not in FACTOR_COLUMNS]

    plot_ts = monthly_betas.select(["date", *from_model])
    if extra:
        supplemental = build_extra_plot_factor_betas(extra, parquet_path).collect()
        plot_ts = plot_ts.join(supplemental, on="date", how="full")

    plot_ts = plot_ts.sort("date").select(
        "date",
        *[
            pl.col(f)
            .rolling_mean(window_size=trail_months, min_samples=trail_months)
            .alias(f)
            for f in PLOT_FACTORS
        ],
    )
    return plot_ts


def main(*, include_stats: bool = INCLUDE_STATS) -> pl.DataFrame:
    """Run Barra_frets WLS and write monthly betas to ``OUTPUT_PATH``."""
    results = build_lazy_wls_plan(include_stats=include_stats).collect()
    flat = results.unnest("betas")
    flat.write_parquet(OUTPUT_PATH)
    return flat


if __name__ == "__main__":
    df = main()
    print(f"Wrote {df.height:,} monthly regressions to {OUTPUT_PATH.resolve()}")
