import os
import numpy as np
import json
import pickle
import torch
import matplotlib.pyplot as plt

file_dir = 'vis_search'
num_test = 30
num_iter = 50
search_space_size_file = os.path.join(file_dir, 'search_space_size.pkl')
total_space_size_file = os.path.join(file_dir, 'total_space_size.pkl')
exp_file = os.path.join(file_dir, 'T_1.json')
convergence_file = os.path.join(file_dir, 'convergence.pkl')
method_types = ['MPPI_BF', 'CROWN']
model_name = '768'
def main():
    if not os.path.exists(search_space_size_file) or not os.path.exists(total_space_size_file):
        process_bab()
    with open(search_space_size_file, 'rb') as f:
        search_space_size_array_agg = pickle.load(f)
    with open(total_space_size_file, 'rb') as f:
        total_space_size_array_agg = pickle.load(f)

    plot_search_space_size(search_space_size_array_agg, 'search_space_size', start_iter=10)
    plot_search_space_size(total_space_size_array_agg, 'total_space_size', start_iter=0)

    if not os.path.exists(convergence_file):
        process_convergence()

    with open(convergence_file, 'rb') as f:
        data = pickle.load(f)
    draw_model_convergence(data["intermediate_results"], data["planned_results"], num_iter, method_types)

def draw_model_convergence(inter_result, planning_results, max_iterations, method_types):
    """
    Draws a subplot for a specific model comparing CROWN with MPPI and MIP.

    Parameters:
    - model_name: Name of the model.
    - inter_results: Intermediate results from CROWN.
    - planning_results: Final results for MPPI and MIP.
    - max_iterations: Maximum number of iterations to plot.
    """
    # Plot CROWN convergence curve with percentiles
    objective_array = np.array(inter_result)
    accumulated_objective = np.sum(objective_array, axis=1)
    mean_objective = np.median(accumulated_objective, axis=0)
    percentile_25 = np.percentile(accumulated_objective, 25, axis=0)
    percentile_75 = np.percentile(accumulated_objective, 75, axis=0)

    iteration_indices = np.arange(0, max_iterations+1)
    color = "#197ab7"
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(iteration_indices, mean_objective, label="CROWN", color=color, linewidth=1.5)
    ax.fill_between(iteration_indices, percentile_25, percentile_75, color=color, edgecolor=color, alpha=0.2)
    ax.plot(iteration_indices, percentile_25, alpha=0.5, color=color, linewidth=0.5)
    ax.plot(iteration_indices, percentile_75, alpha=0.5, color=color, linewidth=0.5)

    # Plot median values for MPPI and MIP
    linestyle = "dashed"
    colors = ["#fbb461", "#b5d66b", "purple", "yellow"]
    i = 0
    for method in method_types:
        if method not in planning_results or method == "CROWN":
            continue
        color = colors[i%len(colors)]
        i += 1
        method_results = planning_results[method]
        method_median = np.median(method_results)
        method_percentile_25 = np.percentile(method_results, 25)
        method_percentile_75 = np.percentile(method_results, 75)

        ax.hlines(method_median, 0, max_iterations, label=f"{method}", color=color, linestyles=linestyle)
        ax.hlines(method_percentile_25, 0, max_iterations, color=color, linestyles=linestyle, alpha=0.5)
        ax.hlines(method_percentile_75, 0, max_iterations, color=color, linestyles=linestyle, alpha=0.5)

    # Set x-axis ticks
    tick_interval = max(max_iterations // 5, 1)
    ax.set_xticks(np.arange(0, max_iterations + 1, tick_interval))
    ax.set_yticks(np.arange(35, 51, 5))
    ax.set_xlim([0, max_iterations])

    ax.legend()
    file_name = os.path.join(file_dir, f'convergence.pdf')
    plt.savefig(file_name)
    print(f"Saved to {file_name}")



def plot_search_space_size(data, name, start_iter=0):
    # [num_test, num_iter+1]
    data = data[:, start_iter:]
    median = np.median(data, axis=0)
    percentile_25 = np.percentile(data, 25, axis=0)
    percentile_75 = np.percentile(data, 75, axis=0)

    # median_color, percentile_color = '#469393', '#d6e2e4'
    median_color, percentile_color = '#197ab7', '#a5cfe3'
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.plot(range(start_iter, num_iter+1), median, color=median_color, label='Median')
    plt.fill_between(range(start_iter, num_iter+1), percentile_25, percentile_75, color=percentile_color, edgecolor=median_color, alpha=0.5, label='25th-75th Percentile')

    # plt.title('Search Space Size Over Iterations')
    # plt.xlabel('Iterations')
    # plt.ylabel('Search Space Size')
    # plt.legend()
    ax.set_xlim([start_iter, num_iter])

    # plt.grid(True)
    file_name = os.path.join(file_dir, f'{name}.pdf')
    plt.savefig(file_name)
    print(f"Saved to {file_name}")

def process_convergence():
    with open(exp_file, "r") as f:
        all_res = json.load(f)
    all_res = all_res[model_name]
    cost_type = 'planned_cost'
    method_types = ['MPPI', 'MPPI_BF', 'CROWN']
    inter_result = []
    method_obj_dict = {method_type: [] for method_type in method_types}
    for i in range(num_test):
        i = str(i)
        method_res = {}
        for method_type in method_types:
            method_res[method_type] = all_res[i][method_type]
            method_obj_dict[method_type].append(method_res[method_type]["result_summary"][cost_type])
            if method_type == 'CROWN':
                inter_result.append(
                    [res["intermediate_best_outputs"] for res in method_res[method_type]["all_res"]]
                )
    # save the data
    with open(convergence_file, 'wb') as f:
        pickle.dump({'intermediate_results': inter_result, 'planned_results': method_obj_dict}, f)

def process_bab():
    # get all .pkl files in the directory
    file_list = [f for f in os.listdir(file_dir) if f.endswith('.pkl')]

    search_space_size_list_agg = []
    total_space_size_list_agg = []
    for bab_file in file_list:
        bab_bounds = []
        with open(os.path.join(file_dir, bab_file), 'rb') as f:
            while True:
                try:
                    bab_bounds.append(pickle.load(f))
                except EOFError:
                    break
        global_x_L = bab_bounds[0]["x_L"]
        global_x_U = bab_bounds[0]["x_U"]
        search_space_size_list = [1] * 2
        total_space_size_list = [1] * 2
        bab_bounds = bab_bounds[1:]
        for bound_dict in bab_bounds:
            search_space_size_list.append(analyze_domain_distribution(global_x_L, global_x_U, bound_dict["x_L"], bound_dict["x_U"]))
            total_space_size_list.append(bound_dict["total_space_size"])

        search_space_size_list_agg.append(search_space_size_list)
        total_space_size_list_agg.append(total_space_size_list)

    search_space_size_list_agg = np.array(search_space_size_list_agg)
    total_space_size_list_agg = np.array(total_space_size_list_agg)
    assert search_space_size_list_agg.shape == (num_test, num_iter+1)
    assert total_space_size_list_agg.shape == (num_test, num_iter+1)
    # save the data
    with open(search_space_size_file, 'wb') as f:
        pickle.dump(search_space_size_list_agg, f)
    with open(total_space_size_file, 'wb') as f:
        pickle.dump(total_space_size_list_agg, f)

def analyze_domain_distribution(global_x_L, global_x_U, x_L, x_U):
    # [num_domain, num_input]
    normed_x_L = (x_L - global_x_L) / (global_x_U - global_x_L)
    normed_x_U = (x_U - global_x_L) / (global_x_U - global_x_L)
    domain_center = (normed_x_L + normed_x_U) / 2
    # sort domain center by distance to origin
    sorted_indices = torch.norm(domain_center, p=1, dim=1).argsort()
    domain_center = domain_center[sorted_indices]
    normed_x_L = normed_x_L[sorted_indices]
    normed_x_U = normed_x_U[sorted_indices]

    covered_space = normed_x_U - normed_x_L
    # [num_domain]
    space_size = covered_space.prod(-1)
    total_space_size = space_size.sum().item()

    return total_space_size


if __name__ == '__main__':
    main()