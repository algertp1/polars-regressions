"""Peak RSS comparison: polars vs numpy vs sklearn WLS (multivariate + univariate)."""

from __future__ import annotations

import gc
import subprocess
import sys
import textwrap

WORKER_HEADER = textwrap.dedent("""
    import gc
    import psutil
    proc = psutil.Process()
    peak = proc.memory_info().rss

    def tick():
        global peak
        peak = max(peak, proc.memory_info().rss)

    def report():
        mb = peak / (1024 ** 2)
        print(f"PEAK_RSS_MB={mb:.1f}", flush=True)
""")

WORKERS: dict[str, str] = {
    "polars_multi": WORKER_HEADER + textwrap.dedent("""
        import polars_ols  # noqa: F401
        from barra_frets import build_lazy_wls_plan
        gc.collect(); tick()
        build_lazy_wls_plan().collect().unnest("betas")
        tick(); report()
    """),
    "numpy_multi": WORKER_HEADER + textwrap.dedent("""
        import numpy as np
        import pandas as pd
        from barra_frets import FACTOR_COLUMNS, PARQUET_PATH
        gc.collect(); tick()
        df = (
            pd.read_parquet(PARQUET_PATH)
            .query("country_gem4 == 'USA'")
            .dropna(subset=["ret", "srisk", *FACTOR_COLUMNS])
        )
        df["regwt"] = 1.0 / df["srisk"] ** 2
        tick()
        for _, grp in df.groupby("date"):
            X = grp[FACTOR_COLUMNS].values
            y = grp["ret"].values
            sw = np.sqrt(grp["regwt"].values)
            np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
            tick()
        report()
    """),
    "sklearn_multi": WORKER_HEADER + textwrap.dedent("""
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from barra_frets import FACTOR_COLUMNS, PARQUET_PATH
        gc.collect(); tick()
        df = (
            pd.read_parquet(PARQUET_PATH)
            .query("country_gem4 == 'USA'")
            .dropna(subset=["ret", "srisk", *FACTOR_COLUMNS])
        )
        df["regwt"] = 1.0 / df["srisk"] ** 2
        tick()
        model = LinearRegression(fit_intercept=False)
        for _, grp in df.groupby("date"):
            model.fit(
                grp[FACTOR_COLUMNS].values,
                grp["ret"].values,
                sample_weight=grp["regwt"].values,
            )
            tick()
        report()
    """),
    "polars_uni": WORKER_HEADER + textwrap.dedent("""
        import polars_ols  # noqa: F401
        from barra_frets import build_lazy_univariate_wls_plan
        gc.collect(); tick()
        build_lazy_univariate_wls_plan().collect()
        tick(); report()
    """),
    "numpy_uni": WORKER_HEADER + textwrap.dedent("""
        import numpy as np
        import pandas as pd
        from barra_frets import FACTOR_COLUMNS, PARQUET_PATH, RISK_FACTORS, INDUSTRY_FACTORS
        gc.collect(); tick()
        df = (
            pd.read_parquet(PARQUET_PATH)
            .query("country_gem4 == 'USA'")
            .dropna(subset=["ret", "srisk", *FACTOR_COLUMNS])
        )
        df["regwt"] = 1.0 / df["srisk"] ** 2
        features = {f: [f, *INDUSTRY_FACTORS] for f in RISK_FACTORS}
        tick()
        for _, grp in df.groupby("date"):
            y = grp["ret"].values
            sw = np.sqrt(grp["regwt"].values)
            yw = y * sw
            for cols in features.values():
                X = grp[cols].values
                np.linalg.lstsq(X * sw[:, None], yw, rcond=None)
            tick()
        report()
    """),
    "sklearn_uni": WORKER_HEADER + textwrap.dedent("""
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from barra_frets import FACTOR_COLUMNS, PARQUET_PATH, RISK_FACTORS, INDUSTRY_FACTORS
        gc.collect(); tick()
        df = (
            pd.read_parquet(PARQUET_PATH)
            .query("country_gem4 == 'USA'")
            .dropna(subset=["ret", "srisk", *FACTOR_COLUMNS])
        )
        df["regwt"] = 1.0 / df["srisk"] ** 2
        features = {f: [f, *INDUSTRY_FACTORS] for f in RISK_FACTORS}
        tick()
        model = LinearRegression(fit_intercept=False)
        for _, grp in df.groupby("date"):
            y = grp["ret"].values
            w = grp["regwt"].values
            for cols in features.values():
                model.fit(grp[cols].values, y, sample_weight=w)
            tick()
        report()
    """),
}


def run_worker(label: str, code: str) -> float:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed ({proc.returncode}):\n{proc.stderr[-2000:]}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("PEAK_RSS_MB="):
            return float(line.split("=", 1)[1])
    raise RuntimeError(f"{label}: no PEAK_RSS_MB in output:\n{proc.stdout}")


def main() -> None:
    print("Peak RSS per approach (fresh process, end-to-end incl. parquet read)\n")
    print(f"{'approach':<18} {'peak RSS':>12}")
    print("-" * 32)

    results: dict[str, float] = {}
    for name, code in WORKERS.items():
        peak = run_worker(name, code)
        results[name] = peak
        print(f"{name:<18} {peak:>10.0f} MB")

    base = results["polars_multi"]
    print(f"\nRelative to polars multivariate ({base:.0f} MB):")
    for name, peak in results.items():
        print(f"  {name:<18} {peak / base:5.2f}x")


if __name__ == "__main__":
    main()
