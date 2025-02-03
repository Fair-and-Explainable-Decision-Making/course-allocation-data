import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from plotly.colors import qualitative

df = pd.read_csv(
    "/scratch3/paula/course-allocation-data/experiments/experiment_results.csv"
)
palette = qualitative.Plotly
# palette = [sns.color_palette("colorblind")[i] for i in [3,0,2]]
palette = [palette[i] for i in [0, 1, 2]]
# faded_palette = [color + '66' for color in palette]
faded_palette = [color + "66" for color in palette]

lex_dict = {}
leximin_list = range(100)
for seed in leximin_list:
    data = np.load(f"experiments/leximin/leximin_{seed}.npz")
    leximin_SD = data["leximin_SD"]
    leximin_RR = data["leximin_RR"]
    leximin_YS = data["leximin_YS"]
    lex_dict[seed] = {"SD": leximin_SD, "RR": leximin_RR, "YS": leximin_YS}

rows = []
all_sizes = range(7)  # Assuming sizes are 0 to 6
for seed, algorithms in lex_dict.items():
    for algo, sizes in algorithms.items():
        freq = pd.Series(sizes).value_counts().sort_index()
        # Ensure all sizes are represented
        freq = freq.reindex(all_sizes, fill_value=0)
        for size, count in freq.items():
            rows.append(
                {"seed": seed, "algorithm": algo, "size": size, "frequency": count}
            )

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
summary["algorithm"] = pd.Categorical(
    summary["algorithm"], categories=algorithm_order, ordered=True
)

# Sort the DataFrame based on the new categorical order
summary = summary.sort_values(by="algorithm")


# Create bar plot
plt.figure(figsize=(12, 3.5))

algorithms = summary["algorithm"].unique()
bar_width = 0.25


# Iterate over each algorithm to plot bars individually
for i, algorithm in enumerate(summary["algorithm"].unique()):
    subset = summary[summary["algorithm"] == algorithm]
    x_positions = [
        size + i * bar_width - (bar_width * (len(algorithms) - 1) / 2)
        for size in subset["size"]
    ]
    plt.bar(
        x=x_positions,
        height=subset["mean_frequency"],
        width=bar_width - 0.02,
        color=faded_palette[i],
        edgecolor=palette[i],
        linewidth=1.5,
    )
# Adding error bars manually
for i, algo in enumerate(summary["algorithm"].unique()):
    algo_data = summary[summary["algorithm"] == algo]
    diff = 0
    if algo == "SD":
        diff = -0.25
    elif algo == "YS":
        diff = 0.25
    plt.errorbar(
        x=algo_data["size"] + diff,
        y=algo_data["mean_frequency"],
        yerr=algo_data["std_frequency"],
        fmt="none",
        c=palette[i],
        capsize=5,
        linewidth=1.5,
    )
    mean_freq_ordered = algo_data.sort_values(by="size")["mean_frequency"].values
    # plt.plot([0,1,2,3,4,5,6], mean_freq_ordered, color= palette[i], linewidth = 2.5)


# Labels and legend
# plt.title("Frequency of Sizes by Algorithm")
plt.xlabel("Bundle Size")
plt.ylabel("Mean Frequency")
plt.tight_layout()

def weighted_mean_std(df):
    mean_size = np.average(df["size"], weights=df["frequency"])
    variance = np.average((df["size"] - mean_size) ** 2, weights=df["frequency"])
    std_size = np.sqrt(variance)
    return pd.Series({"mean_size": mean_size, "std_size": std_size})

result = df.groupby("algorithm").apply(weighted_mean_std).reset_index()





rr_stats = result[result["algorithm"] == "SD"][["mean_size", "std_size"]].values.flatten()
mean_rr, std_rr = rr_stats  # Unpack the values
text = ' μ={:.2f}\n σ={:.2f}'.format(mean_rr, std_rr )
plt.text(0.65, 400, text, color='k', 
        bbox=dict(facecolor='none', edgecolor=palette[0], boxstyle='round'))


rr_stats = result[result["algorithm"] == "RR"][["mean_size", "std_size"]].values.flatten()
mean_rr, std_rr = rr_stats  # Unpack the values
text = ' μ={:.2f}\n σ={:.2f}'.format(mean_rr, std_rr )
plt.text(1.65, 600, text, color='k', 
        bbox=dict(facecolor='none', edgecolor=palette[1], boxstyle='round'))

rr_stats = result[result["algorithm"] == "YS"][["mean_size", "std_size"]].values.flatten()
mean_rr, std_rr = rr_stats  # Unpack the values
text = ' μ={:.2f}\n σ={:.2f}'.format(mean_rr, std_rr )
plt.text(4.65, 600, text, color='k', 
        bbox=dict(facecolor='none', edgecolor=palette[2], boxstyle='round'))


plt.ylim([0,1000])


plt.savefig(f"./experiments/figs/hist.jpg", dpi=300)
plt.savefig(f"./experiments/figs/hist.pdf", format="pdf", dpi=300)
