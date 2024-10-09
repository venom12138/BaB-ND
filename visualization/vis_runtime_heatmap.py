import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'

# Data from earlier context
data_with_status = {
    "Horizon\\Model size": [1, 2, 3, 4, 5],
    232: [0.0921, 0.1837, 0.3040, 0.4493, 0.7687],
    392: [0.1215, 0.4039, 1.0616, 6.3698, 74.1069],
    584: [0.1447, 0.4716, 1.5824, 36.0765, 'Subopt'],
    1064: [0.2490, 1.8167, 122.7804, 'Subopt', 'Subopt'],
    1672: [0.3037, 3.5723, 195.3746, 'Subopt', 'Fail'],
    3272: [0.9940, 176.8832, 'Subopt', 'Fail', 'Fail'],
    5384: [0.9856, 'Subopt', 'Subopt', 'Fail', 'Fail']
}

# Convert to DataFrame and set index
df_status = pd.DataFrame(data_with_status).set_index("Horizon\\Model size")

# Replace 'Subopt' and 'Fail' with very high values to ensure they appear at the end of the spectrum
df_numeric = df_status.replace({
    'Subopt': 300,  # Placeholder for Suboptimal
    'Fail': 500  # Placeholder for Fail
})

# Rank the data and normalize
ranked_values = df_numeric.stack().rank(method='dense').unstack()  # Rank data
max_rank = 24
ranked_mask = ranked_values < max_rank
# max_numeric_rank = ranked_values[ranked_mask].max().max()
ranked_values_normalized = ranked_values.copy()
ranked_values_normalized[ranked_mask] = ranked_values[ranked_mask] / max_rank * 0.8
ranked_values_normalized.replace({24: 0.9, 25: 1.0}, inplace=True)
print(ranked_values_normalized)
# Using 'Spectral_r' diverging palette
palette = sns.color_palette("Spectral_r", as_cmap=True, n_colors=31)

# Plotting the heatmap with ranked colors
plt.figure(figsize=(10, 6))
ax = sns.heatmap(ranked_values_normalized, annot=df_status, cmap=palette,
                 cbar_kws={"label": "", "ticks": []},
                 linewidths=0.5, linecolor='white', annot_kws=dict(size=14),
                 fmt=''
                 )
ax.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
ax.xaxis.set_visible(False)  # Hide x-axis
ax.yaxis.set_visible(False)  # Hide y-axis

plt.savefig('heatmap.pdf')
