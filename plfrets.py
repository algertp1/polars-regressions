"""Monthly Barra factor WLS — lazy Polars reference script."""

from pathlib import Path

import polars as pl
import polars_ols  # noqa: F401 — registers .least_squares namespace

PARQUET_PATH = Path("parquet_files/fexp_panel.parquet")
OUTPUT_PATH = Path("parquet_files/fexp_wls_betas.parquet")

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

FACTOR_COLUMNS = RISK_FACTORS + INDUSTRY_FACTORS


def wls_expr(mode: str) -> pl.Expr:
    return pl.col("ret").least_squares.wls(
        *[pl.col(name) for name in FACTOR_COLUMNS],
        sample_weights=pl.col("regwt"),
        add_intercept=False,
        null_policy="drop",
        solve_method="svd",
        mode=mode,
    )


def build_lazy_wls_plan(parquet_path: Path = PARQUET_PATH) -> pl.LazyFrame:
    weighted_mean_ret = (pl.col("ret") * pl.col("regwt")).sum() / pl.col("regwt").sum()

    return (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("country_gem4") == "USA")
        .with_columns((1.0 / pl.col("srisk").pow(2)).alias("regwt"))
        .group_by("date")
        .agg(
            pl.len().alias("n_obs"),
            wls_expr("coefficients").alias("betas"),
            (pl.col("regwt") * wls_expr("residuals").pow(2)).sum().alias("sse"),
            (pl.col("regwt") * (pl.col("ret") - weighted_mean_ret).pow(2))
            .sum()
            .alias("tss"),
        )
        .with_columns((1.0 - pl.col("sse") / pl.col("tss")).alias("r2"))
        .sort("date")
    )


def main() -> pl.DataFrame:
    results = build_lazy_wls_plan().collect()
    flat = results.unnest("betas")
    flat.write_parquet(OUTPUT_PATH)
    return flat


if __name__ == "__main__":
    df = main()
    print(f"Wrote {df.height:,} monthly regressions to {OUTPUT_PATH.resolve()}")
