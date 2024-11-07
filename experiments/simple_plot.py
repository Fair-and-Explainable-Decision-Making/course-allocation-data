import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('experiments/experiment_results.csv')

# Filter for the algorithms of interest
algorithms = ['YS', 'RR', 'ILP', 'SD']
filtered_data = data[data['alg'].isin(algorithms)]

# Group by `alg` and `k`, then calculate min, max, and mean for each combination
grouped = filtered_data.groupby(['alg', 'k']).agg({
    'seats': ['min', 'max', 'mean']
}).reset_index()

# Rename columns for easier access
grouped.columns = ['alg', 'k', 'seats_min', 'seats_max', 'seats_mean']

# Plotting
plt.figure(figsize=(10, 6))

# Plot for each algorithm individually
for index,alg in enumerate(grouped['alg'].unique()):
    alg_data = grouped[grouped['alg'] == alg]
    
    # Plot error bars
    plt.errorbar(alg_data['k'], alg_data['seats_mean'], 
                 yerr=[alg_data['seats_mean'] - alg_data['seats_min'], alg_data['seats_max'] - alg_data['seats_mean']], 
                 fmt='o', capsize=5, label=alg)
    
    # Connect mean points with a line
    plt.plot(alg_data['k'], alg_data['seats_mean'], linestyle='-', marker='o', color=f'C{index}')

plt.xlabel('k')
plt.ylabel('Seats')
# plt.title('Seats per k with Min, Max, and Mean Error Bars for YS, RR, ILP, SD Algorithms')
plt.legend(title='Algorithm')



plt.savefig(f"experiments/seat_errorbars.png", bbox_inches="tight", dpi=300)

plt.close()



# Group by `alg` and `k`, then calculate min, max, and mean for each combination
grouped = filtered_data.groupby(['alg', 'k']).agg({
    'zeros': ['min', 'max', 'mean']
}).reset_index()

# Rename columns for easier access
grouped.columns = ['alg', 'k', 'zeros_min', 'zeros_max', 'zeros_mean']

# Plotting
plt.figure(figsize=(10, 6))

# Plot for each algorithm individually
for index,alg in enumerate(grouped['alg'].unique()):
    alg_data = grouped[grouped['alg'] == alg]
    
    # Plot error bars
    plt.errorbar(alg_data['k'], alg_data['zeros_mean'], 
                 yerr=[alg_data['zeros_mean'] - alg_data['zeros_min'], alg_data['zeros_max'] - alg_data['zeros_mean']], 
                 fmt='o', capsize=5, label=alg)
    
    # Connect mean points with a line
    plt.plot(alg_data['k'], alg_data['zeros_mean'], linestyle='-', marker='o', color=f'C{index}')

plt.xlabel('k')
plt.ylabel('Zeroes')
# plt.title('Seats per k with Min, Max, and Mean Error Bars for YS, RR, ILP, SD Algorithms')
plt.legend(title='Algorithm')



plt.savefig(f"experiments/zeros_errorbars.png", bbox_inches="tight", dpi=300)

plt.close()