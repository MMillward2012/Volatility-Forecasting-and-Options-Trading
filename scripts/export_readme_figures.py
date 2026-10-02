"""Export the three README figures from saved diagnostics and frozen M2 checks."""

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forecast_robustness import (
    build_robustness_outcomes,
    cumulative_m2_gain,
    fit_static_m2,
    forecast_static_m2,
    headline_30d_predictor_dates,
    score_static_m2,
)
from src.forecasting import build_locked_feature_panel
from src.time_series import load_skew_metrics


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"


def export_existing_surface_figure():
    notebook = json.loads((ROOT / "notebooks" / "skew_metrics_diagnostics.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "b09d325c")
    images = [output["data"]["image/png"] for output in cell["outputs"]
              if "image/png" in output.get("data", {})]
    if len(images) != 1:
        raise ValueError("Expected one saved repaired 30/60/90-day smile plot.")
    FIGURES.joinpath("repaired_smiles_30_60_90.png").write_bytes(
        base64.b64decode("".join(images[0]))
    )


def scored_m2(panel, spx_prices, outcomes, state, horizon, headline_dates):
    fitted = fit_static_m2(panel, spx_prices, state, horizon)
    predictions = forecast_static_m2(fitted, panel, spx_prices)
    if (state, horizon) == ("skew_30", 5):
        predictions = predictions.loc[predictions.quote_date.isin(headline_dates)]
    summary, common = score_static_m2(predictions, outcomes, fitted["target"])
    return summary, common


def draw_figures():
    metrics = load_skew_metrics(ROOT / "data" / "processed" / "spx_skew_metrics_2023_2025.csv")
    spx_prices = pd.read_csv(ROOT / "data" / "raw" / "spx_security_prices.csv")
    vix_prices = pd.read_csv(ROOT / "data" / "raw" / "VIX_security_prices.csv")
    panel = build_locked_feature_panel(metrics, spx_prices, vix_prices)
    outcomes = build_robustness_outcomes(panel, spx_prices)
    headline_dates = headline_30d_predictor_dates(panel, spx_prices)

    results = {
        (state, horizon): scored_m2(panel, spx_prices, outcomes, state, horizon, headline_dates)
        for state in ("skew_30", "skew_30_60") for horizon in (1, 5, 10)
    }
    frozen = {
        "skew_30": (0.1012, 0.0922),
        "skew_30_60": (0.0544, 0.0492),
    }
    for state, (m0_rmse, m2_rmse) in frozen.items():
        summary, _ = results[state, 5]
        if (summary["n"] != 158 or not np.isclose(summary["rmse_m0"], m0_rmse, atol=5e-5)
                or not np.isclose(summary["rmse_m2"], m2_rmse, atol=5e-5)):
            raise ValueError("The saved 2025 M0/M2 checkpoint was not reproduced.")

    titles = {"skew_30": "30-day downside skew", "skew_30_60": "30–60-day skew spread"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, state in zip(axes, titles):
        summary, common = results[state, 5]
        gain = cumulative_m2_gain(common)
        ax.plot(gain.quote_date, gain.gain, color="#165a72", lw=2)
        ax.axhline(0, color="#777777", lw=.8)
        ax.set(title=titles[state], xlabel="2025 forecast origin",
               ylabel="Cumulative M0 − M2 squared loss")
        ax.text(.02, .96, f"158 dates  ·  OOS R² {summary['r2_vs_m0']:.3f}",
                transform=ax.transAxes, va="top", color="#333333")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(alpha=.18)
    fig.suptitle("Locked 2025 confirmation: simple mean reversion vs persistence", fontsize=13)
    fig.savefig(FIGURES / "locked_2025_m2_gain.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    colors = {1: "#808080", 5: "#165a72", 10: "#b97427"}
    for ax, state in zip(axes, titles):
        for horizon in (1, 5, 10):
            _, common = results[state, horizon]
            gain = cumulative_m2_gain(common)
            ax.plot(gain.quote_date, gain.gain, label=f"{horizon} sessions",
                    color=colors[horizon], lw=2 if horizon == 5 else 1.5)
        ax.axhline(0, color="#777777", lw=.8)
        ax.set(title=titles[state], xlabel="2025 forecast origin",
               ylabel="Cumulative M0 − M2 squared loss")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(alpha=.18)
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Horizon robustness: frozen M2 vs persistence", fontsize=13)
    fig.savefig(FIGURES / "horizon_robustness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    export_existing_surface_figure()
    draw_figures()
