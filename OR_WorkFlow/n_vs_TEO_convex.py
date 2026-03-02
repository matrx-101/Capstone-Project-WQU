import pandas as pd
import matplotlib.pyplot as plt

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

# Convex models only
# Convex models only (renamed for academic clarity)
convex_models = {
    "DenseQP": ("black", "o", "-", "Classical Least Squares"),
    "L1_Soft_Sparse": ("dimgray", "s", "--", "L1 Regularized"),
    "L2_QP_Dense": ("gray", "^", "-.", "L2 Regularized")
}
results = []

for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)
    N = dimension_map.get(sheet)

    for model_name in convex_models.keys():
        row = df[df["Model"] == model_name]
        if not row.empty:
            results.append({
                "N": N,
                "Method": model_name,
                "TE_O": row["TE_O"].values[0]
            })

plot_df = pd.DataFrame(results)

plt.figure()

plt.figure()

for model_key, (color, marker, linestyle, display_name) in convex_models.items():
    subset = plot_df[plot_df["Method"] == model_key].sort_values("N")
    if not subset.empty:
        plt.plot(
            subset["N"],
            subset["TE_O"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=display_name
        )

plt.xlabel("Dimension (N)")
plt.ylabel("Out-of-Sample Tracking Error (TE_O)")
plt.title("Convex Baseline Models — TE_O vs Dimension")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()