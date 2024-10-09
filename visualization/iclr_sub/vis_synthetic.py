import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
mpl.rc('font', family='Arial')
import copy

def plot_all_runs_with_percentiles(input_file, plot_file_name):
    # Load the data
    input_file = "optimization_results.pkl"
    with open(input_file, 'rb') as f:
        final_dict = pickle.load(f)
    input_file = "optimization_results_new.pkl"
    with open(input_file, 'rb') as f:
        final_dict_new = pickle.load(f)
    # Combine the two dictionaries
    for seed in final_dict_new:
        final_dict[seed] = final_dict_new[seed]
    # Get all the action dimensions from one of the final_dict entries (assuming all have the same keys)
    action_dims = list(final_dict[0].keys())

    # Initialize dictionaries to store aggregated data
    aggregated_results = {
        "GD": {action_dim: [] for action_dim in action_dims},
        "MPPI": {action_dim: [] for action_dim in action_dims},
        "CEM": {action_dim: [] for action_dim in action_dims},
        "Ours": {action_dim: [] for action_dim in action_dims}
    }
    aggregated_results_opt = copy.deepcopy(aggregated_results)

    # Collect data across all seeds for each method and action dimension
    for seed in final_dict:
        for action_dim in action_dims:
            for method in aggregated_results:
                aggregated_results[method][action_dim].append(final_dict[seed][action_dim][method]['cost'])
                aggregated_results_opt[method][action_dim].append(final_dict[seed][action_dim][method]['num_opt'])

    # Prepare to plot the results

    font_size = 18

    for i, data_to_draw in enumerate([aggregated_results, aggregated_results_opt]):
        plt.figure(figsize=(6, 6))
        if i == 0:
            # plot the optimal values y = -1.9803d
            plt.plot(action_dims, [-1.9803*d for d in action_dims], label="Optimal", linestyle='--', color='black', linewidth=4)
        for method in data_to_draw:
            medians = []
            percentiles_25 = []
            percentiles_75 = []

            # Calculate statistics for each action dimension
            for action_dim in action_dims:
                results = np.array(data_to_draw[method][action_dim])
                medians.append(np.median(results, axis=0))
                percentiles_25.append(np.percentile(results, 25, axis=0))
                percentiles_75.append(np.percentile(results, 75, axis=0))

            # Convert lists to numpy arrays for easier plotting
            medians = np.array(medians)
            percentiles_25 = np.array(percentiles_25)
            percentiles_75 = np.array(percentiles_75)

            # Plot the median
            plt.plot(action_dims, medians, label=f"{method}", linewidth=2)

            # Fill the area between 25th and 75th percentiles
            plt.fill_between(action_dims, percentiles_25, percentiles_75, alpha=0.2)

        plt.legend(loc='best', fontsize=font_size)

        plt.xticks(ticks=range(0, 101, 25), fontsize = font_size)
        if i == 0:
            plt.yticks(ticks=range(-200, 1, 50), fontsize = font_size)
            sub_plot_file_name = plot_file_name.replace(".pdf", "_cost.pdf")
        else:
            plt.yticks(ticks=range(0, 101, 25), fontsize = font_size)
            sub_plot_file_name = plot_file_name.replace(".pdf", "_opt.pdf")
        plt.savefig(sub_plot_file_name)
        plt.close()
        print(f"Plot saved to {sub_plot_file_name}")

if __name__ == "__main__":
    input_file = "optimization_results.pkl"
    plot_file_name = "plot.pdf"
    plot_all_runs_with_percentiles(input_file, plot_file_name)