import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from plotly.colors import qualitative


palette = qualitative.Plotly
# palette = [sns.color_palette("colorblind")[i] for i in [3,0,2]]
palette = [palette[i] for i in [0, 1, 2, 4]]
faded_palette = [color + "66" for color in palette]

df = pd.read_csv(
    "/scratch3/paula/course-allocation-data/experiments/time_experiment_results.csv"
)
# Get unique values of NUM_STUDENTS and algorithms
# num_students_values = sorted(df["NUM_STUDENTS"].unique())
num_students_values = [100,500,1000,1500,2000,3000]
algorithms = ["SD", "RR", "YS"]

plt.figure(figsize=(9, 2.2))
for i,alg in enumerate(algorithms):
    runtime_values=[]   
    averages = []
    for num_students_value in num_students_values:
        filtered_df = df[(df["NUM_STUDENTS"] == num_students_value) & (df["alg"] == algorithms[i])]
        # print(filtered_df)
        filtered_df["runtime"]=filtered_df["runtime"].astype('float')
        runtime_values.append(filtered_df["runtime"].values)
        averages.append(np.mean(filtered_df["runtime"].values))
    
    plt.plot(num_students_values, averages, color=palette[i], marker="o")
    flierprops = dict(markeredgecolor=palette[i])
    box=plt.boxplot(runtime_values, positions=num_students_values, patch_artist=True, widths=70, flierprops=flierprops)
    for whisker, cap in zip(
        box["whiskers"],
        box["caps"],
    ):
        whisker.set_color(palette[i]) # Set whisker color
        whisker.set_linewidth(1.5)  # Make whiskers thicker
        cap.set_color(palette[i])  # Set cap color
        cap.set_linewidth(1.5) 
    plt.xticks([])
    # Customize the boxes
    for patch in (box["boxes"]):
        patch.set_facecolor(faded_palette[i])  
        patch.set_edgecolor(palette[i])  # Set vibrant edge color (no alpha)
        patch.set_linewidth(1.5)  # Edge thickness
        # patch.set_alpha(0.4)  
        # Customize the median line
    for median in (box["medians"]):
        median.set_color(palette[i])  # Set median line color
        median.set_linewidth(1.5)  # Make the line thicker
    for flier in box['fliers']:
        flier.set_color(palette[i])
        


num_students_values = [100,200,300,400,500]
#ILP separately    
runtime_values=[]   
averages = []
for num_students_value in num_students_values:
    filtered_df = df[(df["NUM_STUDENTS"] == num_students_value) & (df["alg"] == "ILP")]
    # print(filtered_df)
    filtered_df["runtime"]=filtered_df["runtime"].astype('float')
    
    runtime_values.append(filtered_df["runtime"].values)
    averages.append(np.mean(filtered_df["runtime"].values))

plt.plot(num_students_values, averages, color=palette[3], marker="o")
flierprops = dict(markeredgecolor=palette[3])
box=plt.boxplot(runtime_values, positions=num_students_values, patch_artist=True, widths=70, flierprops=flierprops)
for whisker, cap in zip(
    box["whiskers"],
    box["caps"],
):
    whisker.set_color(palette[3]) # Set whisker color
    whisker.set_linewidth(1.5)  # Make whiskers thicker
    cap.set_color(palette[3])  # Set cap color
    cap.set_linewidth(1.5) 

# Customize the boxes
for patch in (box["boxes"]):
    patch.set_facecolor(faded_palette[3])  
    patch.set_edgecolor(palette[3])  # Set vibrant edge color (no alpha)
    patch.set_linewidth(1.5)  # Edge thickness
    # patch.set_alpha(0.4)  
for median in (box["medians"]):
    median.set_color(palette[3])  # Set median line color
    median.set_linewidth(1.5)  # Make the line thicker
# plt.xticks([100,250,500,1000, 1500, 2000, 2500, 3000])
plt.xticks([])
    
legend_elements = [
    Patch(
        facecolor=faded_palette[0], edgecolor=palette[0], label="SD"
    ),
    Patch(facecolor=faded_palette[1], edgecolor=palette[1], label="RR"),
    Patch(facecolor=faded_palette[2], edgecolor=palette[2], label="YS"),
    Patch(facecolor=faded_palette[3], edgecolor=palette[3], label="ILP"),
]
plt.legend(
    handles=legend_elements,
    ncol=2,
    # bbox_to_anchor=(-0.049, 0.42),
    loc="upper center",
    fontsize=10.5,
)

plt.xticks([100, 500, 1000, 1500, 2000, 3000])


plt.xlabel("Number of Students")
plt.ylabel("Runtime (s)")
plt.tight_layout()
plt.savefig(f"./experiments/figs/time.jpg", dpi=300)
plt.savefig(f"./experiments/figs/time.pdf", format="pdf", dpi=300)

