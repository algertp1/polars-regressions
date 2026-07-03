import polars as pl
import polars_ols as pls  # Registers the .least_squares namespace

# 1. Lazily scan the parquet file (no data is loaded yet)
lazy_plan = (
    pl.scan_parquet("your_data.parquet")
    # 2. Apply your row filter early (Predicate Pushdown)
    .filter(pl.col("your_filter_column") == "some_value")
    # 3. Compute your weights lazily on the fly
    .with_columns(
        (1.0 / pl.col("idio_variance")).alias("wts")
    )
    # 4. Group by your temporal dimension
    .group_by("month")
    .agg(
        # Extract the coefficients per month
        pl.col("y").least_squares.wls(
            pl.col("x1"), pl.col("x2"), pl.col("x3"), # Add your independent variables
            sample_weights=pl.col("wts"),
            add_intercept=True,
            mode="coefficients" # Returns a struct of the beta coefficients
        ).alias("betas"),
        
        # Extract the R-squared or Fit Metrics per month
        pl.col("y").least_squares.wls(
            pl.col("x1"), pl.col("x2"), pl.col("x3"),
            sample_weights=pl.col("wts"),
            add_intercept=True,
            mode="summary" # Returns a struct containing R2, SSE, TSS, etc.
        ).alias("stats")
    )
)

# 5. Execute the entire optimized query graph at once
results_df = lazy_plan.collect()

# 6 convert to normal df
final_flat_df = (
    results_df
    .unnest("betas")
    .unnest("stats")
    .sort("month")
)

shape(final_flat_df)