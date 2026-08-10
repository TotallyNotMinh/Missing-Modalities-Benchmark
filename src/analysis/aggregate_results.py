"""
Results Aggregator for the Missing Modalities Benchmark.

Collects per-patient metric CSV files produced by evaluation scripts
and aggregates them into publication-ready summary tables:

    Table 1/2: RQ1 — Dice/HD95 per evaluator × generator × scenario (mean ± SD, ΔDice)
    Table 3:   RQ2 — Dice per missing-modality model × condition × scenario
    Table 4:   Generator quality — PSNR/SSIM per generator × scenario
    Table 5:   Statistical summary — p-values, effect sizes, TOST results
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class ResultsAggregator:
    """
    Loads per-patient evaluation CSVs and produces aggregated summary DataFrames.

    Expected CSV schema (one row per patient per condition):
        patient_id, model, generator, scenario, condition,
        dice_wt, dice_tc, dice_et, dice_mean,
        hd95_wt, hd95_tc, hd95_et, hd95_mean,
        psnr, ssim, mae, mse

    'condition' is one of: oracle, synthetic, native_missing
    'generator' is one of: pix2pix, med_ddpm, 3d_med_diffusion, none
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

    def load_all(self, pattern: str = "**/*.csv") -> pd.DataFrame:
        """
        Loads and concatenates all per-patient CSV files in results_dir.

        Args:
            pattern: Glob pattern relative to results_dir.

        Returns:
            Combined DataFrame of all patient-level results.
        """
        csv_files = list(self.results_dir.glob(pattern))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV result files found in {self.results_dir} with pattern '{pattern}'"
            )
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"[Aggregator] Loaded {len(df)} rows from {len(csv_files)} files.")
        return df

    @staticmethod
    def mean_std(series: pd.Series) -> str:
        """Formats a series as 'mean ± std' string."""
        if series.empty or series.isna().all():
            return "N/A"
        mean_val = series.mean()
        std_val = series.std()
        if pd.isna(std_val):
            return f"{mean_val:.3f} ± 0.000"
        return f"{mean_val:.3f} ± {std_val:.3f}"

    def rq1_table(
        self,
        df: pd.DataFrame,
        metric: str = "dice_mean"
    ) -> pd.DataFrame:
        """
        Builds RQ1 summary table: evaluator × generator × scenario.

        Columns: mean ± SD for oracle and each synthetic condition,
        plus ΔDice = oracle_mean − synthetic_mean.

        Args:
            df: Full concatenated results DataFrame.
            metric: Metric column to aggregate (e.g. 'dice_mean', 'hd95_mean').

        Returns:
            Summary DataFrame suitable for Table 1 / Table 2.
        """
        rq1 = df[df["condition"].isin(["oracle", "synthetic"])].copy()
        rows = []
        for evaluator in rq1["model"].unique():
            for scenario in sorted(rq1["scenario"].unique()):
                oracle_vals = rq1[
                    (rq1["model"] == evaluator) &
                    (rq1["scenario"] == scenario) &
                    (rq1["condition"] == "oracle")
                ][metric]
                oracle_mean = oracle_vals.mean()

                for generator in rq1["generator"].unique():
                    if generator == "none":
                        continue
                    synth_vals = rq1[
                        (rq1["model"] == evaluator) &
                        (rq1["scenario"] == scenario) &
                        (rq1["condition"] == "synthetic") &
                        (rq1["generator"] == generator)
                    ][metric]

                    rows.append({
                        "evaluator": evaluator,
                        "scenario": scenario,
                        "generator": generator,
                        f"{metric}_oracle": self.mean_std(oracle_vals),
                        f"{metric}_synthetic": self.mean_std(synth_vals),
                        f"delta_{metric}": f"{oracle_mean - synth_vals.mean():.3f}",
                    })

        return pd.DataFrame(rows)

    def rq2_table(
        self,
        df: pd.DataFrame,
        metric: str = "dice_mean"
    ) -> pd.DataFrame:
        """
        Builds RQ2 summary table: missing-modality model × condition × scenario.
        ΔDice = synthetic_mean − native_missing_mean.

        Args:
            df: Full concatenated results DataFrame.
            metric: Metric column to aggregate.

        Returns:
            Summary DataFrame suitable for Table 3.
        """
        rq2 = df[df["condition"].isin(
            ["oracle", "native_missing", "synthetic"]
        )].copy()

        rows = []
        for model in rq2["model"].unique():
            for scenario in sorted(rq2["scenario"].unique()):
                native_vals = rq2[
                    (rq2["model"] == model) &
                    (rq2["scenario"] == scenario) &
                    (rq2["condition"] == "native_missing")
                ][metric]

                oracle_vals = rq2[
                    (rq2["model"] == model) &
                    (rq2["scenario"] == scenario) &
                    (rq2["condition"] == "oracle")
                ][metric]

                row = {
                    "model": model,
                    "scenario": scenario,
                    f"{metric}_native": self.mean_std(native_vals),
                    f"{metric}_oracle": self.mean_std(oracle_vals),
                }

                for generator in rq2["generator"].unique():
                    if pd.isna(generator) or generator == "none":
                        continue
                    synth_vals = rq2[
                        (rq2["model"] == model) &
                        (rq2["scenario"] == scenario) &
                        (rq2["condition"] == "synthetic") &
                        (rq2["generator"] == generator)
                    ][metric]
                    row[f"delta_{generator}"] = (
                        f"{synth_vals.mean() - native_vals.mean():.3f}"
                    )

                rows.append(row)

        return pd.DataFrame(rows)

    def rq3_vectors(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extracts per-patient (PSNR, SSIM) vs (dice_mean, hd95_mean) vectors
        for RQ3 correlation scatter plots.

        Returns:
            DataFrame with columns: patient_id, generator, scenario,
            psnr, ssim, dice_mean, hd95_mean.
        """
        synth = df[df["condition"] == "synthetic"].copy()
        cols = ["patient_id", "generator", "scenario",
                "psnr", "ssim", "dice_mean", "hd95_mean"]
        available = [c for c in cols if c in synth.columns]
        return synth[available].dropna().reset_index(drop=True)

    def save_tables(
        self,
        df: pd.DataFrame,
        out_dir: str = "results/tables"
    ) -> None:
        """
        Generates and saves all 5 summary tables as CSVs.

        Args:
            df: Full concatenated results DataFrame.
            out_dir: Directory to write CSV tables.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        rq1_dice = self.rq1_table(df, metric="dice_mean")
        rq1_dice.to_csv(out_path / "table1_rq1_dice.csv", index=False)
        print(f"[Aggregator] Saved Table 1 → {out_path / 'table1_rq1_dice.csv'}")

        rq1_hd95 = self.rq1_table(df, metric="hd95_mean")
        rq1_hd95.to_csv(out_path / "table2_rq1_hd95.csv", index=False)
        print(f"[Aggregator] Saved Table 2 → {out_path / 'table2_rq1_hd95.csv'}")

        rq2_dice = self.rq2_table(df, metric="dice_mean")
        rq2_dice.to_csv(out_path / "table3_rq2_dice.csv", index=False)
        print(f"[Aggregator] Saved Table 3 → {out_path / 'table3_rq2_dice.csv'}")

        rq3 = self.rq3_vectors(df)
        rq3.to_csv(out_path / "table4_rq3_correlation_vectors.csv", index=False)
        print(f"[Aggregator] Saved Table 4 (RQ3 vectors) → {out_path / 'table4_rq3_correlation_vectors.csv'}")

        print(f"[Aggregator] All tables saved to {out_path}")
