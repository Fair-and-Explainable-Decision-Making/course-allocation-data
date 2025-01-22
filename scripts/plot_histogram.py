import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from plotly.colors import qualitative

df = pd.read_csv('/scratch3/paula/course-allocation-data/experiments/experiment_results.csv')
palette = qualitative.Plotly    
# palette = [sns.color_palette("colorblind")[i] for i in [3,0,2]]
palette = [palette[i] for i in [0,1,2]]
# faded_palette = [color + '66' for color in palette]
faded_palette = [color + '66' for color in palette]

lex_dict={}
leximin_list=[*range(38), *range(40,48)]
for seed in leximin_list:
    data=np.load(f"leximin_{seed}.npz")
    leximin_SD = data["leximin_SD"]
    leximin_RR = data["leximin_RR"]
    leximin_YS = data["leximin_YS"]
    lex_dict[seed] = {"SD":leximin_SD,"RR": leximin_RR, "YS": leximin_YS }

rows = []
all_sizes = range(7)  # Assuming sizes are 0 to 6
for seed, algorithms in lex_dict.items():
    for algo, sizes in algorithms.items():
        freq = pd.Series(sizes).value_counts().sort_index()
        # Ensure all sizes are represented
        freq = freq.reindex(all_sizes, fill_value=0)
        for size, count in freq.items():
            rows.append({"seed": seed, "algorithm": algo, "size": size, "frequency": count})

# Convert to DataFrame
df = pd.DataFrame(rows)
print(df)

summary = (
    df.groupby(["algorithm", "size"])
    .agg(mean_frequency=("frequency", "mean"), std_frequency=("frequency", "std"))
    .reset_index()
)
# Define the desired order for the algorithm column
algorithm_order = ["SD", "RR", "YS"]

# Convert the 'algorithm' column to a categorical type with the specified order
summary["algorithm"] = pd.Categorical(summary["algorithm"], categories=algorithm_order, ordered=True)

# Sort the DataFrame based on the new categorical order
summary = summary.sort_values(by="algorithm")


# Create bar plot
plt.figure(figsize=(12, 4))

algorithms = summary["algorithm"].unique()
bar_width = 0.25



# Iterate over each algorithm to plot bars individually
for i, algorithm in enumerate(summary["algorithm"].unique()):
    subset = summary[summary["algorithm"] == algorithm]
    x_positions = [size + i * bar_width - (bar_width * (len(algorithms) - 1) / 2) for size in subset["size"]]
    plt.bar(
        x=x_positions,
        height=subset["mean_frequency"],
        width=bar_width-0.02,
        color=faded_palette[i],
        edgecolor=palette[i],
        linewidth=1.5, 
    )
# Adding error bars manually
for i,algo in enumerate(summary["algorithm"].unique()):
    algo_data = summary[summary["algorithm"] == algo]
    diff=0
    if algo == "SD":
        diff=-0.25
    elif algo =="YS":
        diff=0.25
    plt.errorbar(
        x=algo_data["size"]+diff,
        y=algo_data["mean_frequency"],
        yerr=algo_data["std_frequency"],
        fmt="none",
        c=palette[i],
        capsize=5,
        linewidth=1.5, 
    )

# Labels and legend
# plt.title("Frequency of Sizes by Algorithm")
plt.xlabel("Bundle Size")
plt.ylabel("Mean Frequency")
plt.tight_layout()



# plt.tight_layout()
plt.savefig(f"./experiments/hist.jpg", dpi=300)
plt.savefig(f"./experiments/hist.pdf", format="pdf",dpi=300)
