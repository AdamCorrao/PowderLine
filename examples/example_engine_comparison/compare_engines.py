"""Run the same LaB6 recipe through GSAS-II and easydiffraction and compare.

Usage (from the repo root; the easydiff environment contains both engines):

    pixi run -e easydiff python examples/example_engine_comparison/compare_engines.py

Outputs (written next to this script):
    output/gsasii/           standard engine outputs (GSAS-II)
    output/easydiffraction/  standard engine outputs (easydiffraction)
    output/comparison.csv    side-by-side refined parameters + fit statistics
    output/comparison.png    two-panel observed/calculated/difference plot

Exit status is non-zero if either refinement fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RECIPE_PATH = REPO_ROOT / "examples" / "example_LaB6_easydiff" / "input.json"
OUTPUT_DIR = HERE / "output"

try:
    from powderline.engine import run
except ImportError:  # running outside the pixi env with editable install
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from powderline.engine import run

# Validated categorical palette (dataviz slots 1-2) + neutral ink for observed.
COLOR_OBS = "#555555"
COLOR_CALC = "#2a78d6"
COLOR_DIFF = "#eb6834"

ENGINES = ("gsasii", "easydiffraction")


def run_engine(recipe: dict, engine: str) -> dict:
    out = OUTPUT_DIR / engine
    out.mkdir(parents=True, exist_ok=True)
    kwargs = {"execution_mode": "subprocess"} if engine == "gsasii" else {}
    print(f"\n=== running engine={engine!r} -> {out} ===")
    result = run(recipe, out, engine=engine, **kwargs)
    if not result["success"]:
        print(f"ERROR: engine {engine!r} failed: {result['error']}", file=sys.stderr)
        sys.exit(1)
    return result


def cell_a(result: dict, phase: str) -> tuple[float, float]:
    df = result["unit_cell_data"][phase]
    row = df[df["parameter"] == "cell_a"].iloc[0]
    return float(row["value"]), float(row["esd"])


def comparison_table(results: dict, phase: str) -> pd.DataFrame:
    """Outer-merge refined parameters; prepend fit statistics and cell a."""
    frames = []
    for engine in ENGINES:
        df = results[engine]["refined_parameters"][
            ["parameter_name", "descriptive_name", "category", "value", "esd"]
        ].copy()
        df = df.rename(columns={"value": f"value_{engine}", "esd": f"esd_{engine}"})
        frames.append(df)
    merged = frames[0].merge(
        frames[1], on=["parameter_name", "descriptive_name", "category"], how="outer"
    )

    head_rows = []
    for name in ("rwp", "gof"):
        head_rows.append(
            {
                "parameter_name": name,
                "descriptive_name": {"rwp": "weighted profile R (%)", "gof": "goodness of fit"}[name],
                "category": "fit_statistics",
                **{f"value_{e}": results[e].get(name) for e in ENGINES},
                **{f"esd_{e}": None for e in ENGINES},
            }
        )
    a_row = {
        "parameter_name": "cell_a",
        "descriptive_name": f"{phase} cubic lattice parameter (A)",
        "category": "unit_cell",
    }
    for engine in ENGINES:
        a_row[f"value_{engine}"], a_row[f"esd_{engine}"] = cell_a(results[engine], phase)
    head_rows.append(a_row)

    return pd.concat([pd.DataFrame(head_rows), merged], ignore_index=True)


def plot_comparison(results: dict, fit_range: tuple[float, float], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=150)
    lo, hi = fit_range
    for ax, engine in zip(axes, ENGINES):
        result = results[engine]
        fp = result["fit_profile"]
        m = (fp["two_theta"] >= lo) & (fp["two_theta"] <= hi)
        fp = fp[m]
        diff = fp["y_obs"] - fp["y_calc"]
        offset = float(fp["y_obs"].min()) - 1.1 * float(diff.abs().max())

        ax.plot(fp["two_theta"], fp["y_obs"], "o", ms=2.5, mfc="none",
                mec=COLOR_OBS, mew=0.6, label="observed")
        ax.plot(fp["two_theta"], fp["y_calc"], color=COLOR_CALC, lw=1.2,
                label="calculated")
        ax.plot(fp["two_theta"], diff + offset, color=COLOR_DIFF, lw=1.0,
                label="difference (offset)")
        ax.axhline(offset, color="#c8c8c8", lw=0.8, zorder=0)

        rwp = result.get("rwp")
        title = "GSAS-II" if engine == "gsasii" else "EasyDiffraction"
        ax.set_title(f"{title}   (Rwp = {rwp:.2f}%)", loc="left", fontsize=11)
        ax.set_ylabel("intensity (counts)")
        ax.grid(color="#eeeeee", lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    axes[1].set_xlabel(r"2$\theta$ (degrees)")
    fig.suptitle("LaB6: identical recipe refined by both engines", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    recipe = json.loads(RECIPE_PATH.read_text())
    phase = next(iter(recipe["payload"]["phases"]))
    fit_range = tuple(recipe["payload"]["fit_range"])

    results = {engine: run_engine(recipe, engine) for engine in ENGINES}

    table = comparison_table(results, phase)
    csv_path = OUTPUT_DIR / "comparison.csv"
    table.to_csv(csv_path, index=False, float_format="%.6g")
    png_path = OUTPUT_DIR / "comparison.png"
    plot_comparison(results, fit_range, png_path)

    print("\n=== engine comparison (same recipe) ===")
    for engine in ENGINES:
        a, esd = cell_a(results[engine], phase)
        print(
            f"{engine:>16}:  Rwp = {results[engine].get('rwp'):6.2f}%   "
            f"GoF = {results[engine].get('gof') if results[engine].get('gof') is not None else float('nan'):5.2f}   "
            f"cell a = {a:.5f} +/- {esd:.5f} A"
        )
    print(f"{'NIST SRM 660':>16}:  a = 4.15682 A (reference)")
    print(
        "\nNotes: the Rwp gap is expected -- easydiffraction's pseudo-Voigt profile has\n"
        "no SH/L axial-divergence asymmetry and no background peaks, both of which the\n"
        "GSAS-II model includes. Scale factors use different normalizations and are not\n"
        "comparable between engines; the lattice parameter is the meaningful cross-check.\n"
        f"\nWrote {csv_path}\nWrote {png_path}"
    )


if __name__ == "__main__":
    main()
