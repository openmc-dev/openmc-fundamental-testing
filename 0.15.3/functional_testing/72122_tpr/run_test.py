# # FNG-HCPB Tritirum Production
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

# %%
TALLY = "FNG-HCPB_H3 Tritium activity Pellet"
LIBRARY_CODE = {
    "Experiment": "_exp_-_exp_",
    "MCNP": "_mcnp_-_FENDL 3.2c_",
    "OpenMC": "_openmc_-_FENDL 3.2c_",
}

COMPARISONS = {
    "OpenMC-EXP": ("OpenMC", "Experiment"),
    "MCNP-EXP": ("MCNP", "Experiment"),
    "OpenMC-MCNP": ("OpenMC", "MCNP"),
}
T_LIMIT = 1.95

COLORS = {"Experiment": "black", "MCNP": "tab:blue", "OpenMC": "tab:red"}

JADE_RAW_RESULTS_URL = "https://raw.githubusercontent.com/JADE-V-V/JADE-RAW-RESULTS/main/ROOT/"

# %%
def align_results(results):
    """Align datasets by distance and pellet, not CSV row order."""
    aligned = {}
    for label in LIBRARY_CODE:
        frame = results[label][["Distance", "Pellet", "Value", "Error"]].copy()
        aligned[label] = frame.set_index(
            ["Distance", "Pellet"]
        ).sort_index()
    return aligned

# %%
def load_results(cache_dir=".cache"):
    """Download files for each library and tally if not already present."""
    results = {}
    for label, library in LIBRARY_CODE.items():
        relative_path = f"{library}/FNG-HCPB/{TALLY}.csv"
        path = Path(cache_dir) / relative_path
        if not path.exists():
            url = (
                JADE_RAW_RESULTS_URL + quote(relative_path, safe="/")
            )
            print(f"Downloading {label}: {TALLY}")
            with urlopen(url) as response:
                content = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        results[label] = pd.read_csv(path)
    return align_results(results)

# %%
def plot_tallies(results):
    """Plot response and calculation/experiment ratios at each distance."""
    figures = []
    for distance in results["Experiment"].index.unique("Distance"):
        experiment = results["Experiment"].loc[distance]
        pellets = experiment.index
        fig, (response_ax, ratio_ax) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        for label, frame in results.items():
            data = frame.loc[distance]
            response_ax.errorbar(
                pellets, data["Value"], yerr=data["Value"] * data["Error"],
                marker="o", capsize=3, color=COLORS[label], label=label,
            )
            if label != "Experiment":
                ratio = data["Value"] / experiment["Value"]
                uncertainty = ratio * np.sqrt(
                    data["Error"] ** 2 + experiment["Error"] ** 2
                )
                ratio_ax.errorbar(
                    pellets, ratio, yerr=uncertainty,
                    marker="o", capsize=3, color=COLORS[label], label=label,
                )
        response_ax.set(
            # yscale="log", 
            ylabel="Activity [Bq/g]",
            title=f"{TALLY} - Distance: {distance:g} cm",
        )
        ratio_ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ratio_ax.set(ylabel="C/E", xlabel="Pellet number", xticks=pellets)
        for ax in (response_ax, ratio_ax):
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
        fig.tight_layout()
        plt.savefig(f"72122_{TALLY}_distance_{distance:g}.png".replace(" ", "_"))
        figures.append(fig)
    return figures

# %%
def t_test_calc(results):
    """Compare pellets and append the mean absolute pellet t at each distance."""
    tables = []
    for distance in results["Experiment"].index.unique("Distance"):
        values = pd.DataFrame({
            label: frame.loc[distance, "Value"]
            for label, frame in results.items()
        })
        errors = pd.DataFrame({
            label: frame.loc[distance, "Value"] * frame.loc[distance, "Error"]
            for label, frame in results.items()
        })
        values.index = values.index.map(str)
        errors.index = errors.index.map(str)

        table = pd.DataFrame({
            "Distance [cm]": distance,
            "Pellet": [*values.index, "Average"],
        })
        for comparison, (left, right) in COMPARISONS.items():
            combined_error = np.sqrt(errors[left] ** 2 + errors[right] ** 2)
            if (combined_error == 0).any():
                raise ValueError(
                    f"{comparison} has zero combined uncertainty at {distance:g} cm"
                )
            t = ((values[left] - values[right]) / combined_error).to_numpy()
            t = np.append(t, np.abs(t).mean())
            table[f"{comparison} t"] = t
            table[f"{comparison} pass"] = np.abs(t) < T_LIMIT
        tables.append(table)
    return pd.concat(tables, ignore_index=True)

# %%
def plot_t_heatmap(details, comparison="OpenMC-EXP t", pass_fail=False):
    """Plot absolute pellet t statistics and their descriptive mean."""
    pellets = details["Pellet"].drop_duplicates().tolist()
    heatmap = details.pivot(
        index="Distance [cm]", columns="Pellet", values=comparison,
    ).reindex(columns=pellets).abs()
    fig, ax = plt.subplots(figsize=(12, max(6, len(heatmap) * 0.4)))
    image = ax.imshow(
        heatmap.to_numpy(), cmap="RdYlGn_r", vmin=0, vmax=4, aspect="auto",
    )
    for row in range(len(heatmap)):
        for column in range(len(heatmap.columns)):
            value = heatmap.iat[row, column]
            text = ("PASS" if value < T_LIMIT else "FAIL") if pass_fail else f"{value:.1f}"
            ax.text(column, row, text, ha="center", va="center")
    ax.set_xticks(range(len(heatmap.columns)), labels=heatmap.columns)
    ax.set_yticks(range(len(heatmap)), labels=[f"{d:g}" for d in heatmap.index])
    ax.set(
        xlabel="Pellet", ylabel="Distance [cm]",
        title=(
            f"{comparison}-test{' PASS/FAIL' if pass_fail else ''}"
        ),
    )
    fig.colorbar(image, ax=ax, label=r"$|t|$")
    fig.tight_layout()
    plt.savefig(f"72122_{comparison}_heatmap.png".replace(" ", "_"))
    return fig

# %%
results = load_results(cache_dir=Path.cwd())

# %%
plot_tallies(results)

# %%
details = t_test_calc(results)

for pass_fail in (False, True):
    for comparison in COMPARISONS:
        plot_t_heatmap(details, f"{comparison} t", pass_fail=pass_fail)

# %%
# PASS only when every distance's OpenMC-MCNP t is below T_LIMIT.
average_t = details.loc[details["Pellet"] == "Average", "OpenMC-MCNP t"]
overall_pass = bool(
    (average_t.abs() < T_LIMIT).all()
)
result = 'PASS' if overall_pass else 'FAIL'
with open('results.txt', 'w') as results_file:
    results_file.write(result + '\n')
print(result)


