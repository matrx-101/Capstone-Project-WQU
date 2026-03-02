import pandas as pd
import matplotlib.pyplot as plt
import re

file_path = "or_results.xlsx"
xls = pd.ExcelFile(file_path)

# Map sheet names to real index names
index_names = {
    "indtrack1": "Hang Seng (N = 31)",
    "indtrack2": "DAX (N = 85)",
    "indtrack3": "FTSE (N = 89)",
    "indtrack4": "S&P 100 (N = 98)",
    "indtrack5": "Nikkei (N = 225)",
    "indtrack6": "S&P 500 (N = 457)",
    "indtrack7": "Russell 1000 (N = 1318)",
    "indtrack8": "Russell 2000 (N = 2151)"
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9
})

for sheet in xls.sheet_names:
    
    df = pd.read_excel(xls, sheet_name=sheet)
    
    def extract_k(model):
        match = re.search(r"_K_(\d+)", model)
        return int(match.group(1)) if match else None
    
    df["K"] = df["Model"].apply(extract_k)
    
    title_name = index_names.get(sheet, sheet)

    # GA FAMILY
    ga_models = {
        "GA": ("black", "o", "-"),
        "HGA_QP": ("dimgray", "s", "--"),
        "HGA_L1": ("gray", "^", "-."),
        "HGA_L2": ("darkgray", "d", ":")
    }
    
    plt.figure()
    
    for model_prefix, (color, marker, linestyle) in ga_models.items():
        subset = df[df["Model"].str.startswith(model_prefix)]
        subset = subset.sort_values("K")
        plt.plot(
            subset["K"],
            subset["TE_O"],
            label=model_prefix,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5
        )
    
    plt.xlabel("Cardinality (K)")
    plt.ylabel("Out-of-Sample Tracking Error")
    plt.title(f"{title_name} – GA Family")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # PSO FAMILY
    pso_models = {
        "PSO": ("black", "o", "-"),
        "HPSO_QP": ("dimgray", "s", "--"),
        "HPSO_L1": ("gray", "^", "-."),
        "HPSO_L2": ("darkgray", "d", ":")
    }
    
    plt.figure()
    
    for model_prefix, (color, marker, linestyle) in pso_models.items():
        subset = df[df["Model"].str.startswith(model_prefix)]
        subset = subset.sort_values("K")
        plt.plot(
            subset["K"],
            subset["TE_O"],
            label=model_prefix,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5
        )
    
    plt.xlabel("Cardinality (K)")
    plt.ylabel("Out-of-Sample Tracking Error")
    plt.title(f"{title_name} – PSO Family")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()