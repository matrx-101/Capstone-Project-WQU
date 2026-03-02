import pandas as pd
import matplotlib.pyplot as plt
import re

file_path = "or_results.xlsx"
xls = pd.ExcelFile(file_path)

# Map sheet names to dimensions
dimension_map = {
    "indtrack1": 31,
    "indtrack2": 85,
    "indtrack3": 89,
    "indtrack4": 98,
    "indtrack5": 225,
    "indtrack6": 457,
    "indtrack7": 1318,
    "indtrack8": 2151
}

# Academic plot settings
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9
})

K_values = [5, 10, 20, 40, 50]

ga_models = {
    "GA": ("black", "o", "-"),
    "HGA_QP": ("dimgray", "s", "--"),
    "HGA_L1": ("gray", "^", "-."),
    "HGA_L2": ("darkgray", "d", ":")
}

pso_models = {
    "PSO": ("black", "o", "-"),
    "HPSO_QP": ("dimgray", "s", "--"),
    "HPSO_L1": ("gray", "^", "-."),
    "HPSO_L2": ("darkgray", "d", ":")
}

for K in K_values:

    results = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)

        N = dimension_map.get(sheet)

        for model_prefix in list(ga_models.keys()) + list(pso_models.keys()):
            model_name = f"{model_prefix}_K_{K}"
            row = df[df["Model"] == model_name]

            if not row.empty:
                results.append({
                    "N": N,
                    "Method": model_prefix,
                    "TE_O": row["TE_O"].values[0]
                })

    plot_df = pd.DataFrame(results)

    # GA FAMILY
    plt.figure()

    for model_prefix, (color, marker, linestyle) in ga_models.items():
        subset = plot_df[plot_df["Method"] == model_prefix].sort_values("N")
        if not subset.empty:
            plt.plot(
                subset["N"],
                subset["TE_O"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=model_prefix
            )

    plt.xlabel("Dimension (N)")
    plt.ylabel("Out-of-Sample Tracking Error (TE_O)")
    plt.title(f"GA Family — TE_O vs Dimension (K = {K})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # PSO FAMILY
    plt.figure()

    for model_prefix, (color, marker, linestyle) in pso_models.items():
        subset = plot_df[plot_df["Method"] == model_prefix].sort_values("N")
        if not subset.empty:
            plt.plot(
                subset["N"],
                subset["TE_O"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=model_prefix
            )

    plt.xlabel("Dimension (N)")
    plt.ylabel("Out-of-Sample Tracking Error (TE_O)")
    plt.title(f"PSO Family — TE_O vs Dimension (K = {K})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()