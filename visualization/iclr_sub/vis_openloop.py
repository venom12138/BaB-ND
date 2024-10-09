import os
import sys

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

sys.path.append(os.getcwd())
cwd = os.getcwd()
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib.ticker import MaxNLocator
import re


def main():
    result_dir = '.'
    exp_name_list = ['rope', 'pile', 'L', 'T']
    for exp_name in exp_name_list:
        if 'rope' in exp_name:
            task_name = 'Rope Routing'
        elif 'pile' in exp_name:
            task_name = 'Object Sorting'
        elif 'L' in exp_name:
            task_name = 'Object Merging'
        elif 'T' in exp_name:
            task_name = 'Object Pushing w/ Obs.'
        

        stat = {}
        inter_results = {}
        inter_result = []
        method_types = ['GD_BF', 'MPPI_BF', 'DecentCEM', 'CROWN']
        
        num_test = 20
        cost_type = 'planned_cost'
        method_obj_dict = {method_type: [] for method_type in method_types}

        for method_type in method_types:
            for i in range(num_test):
                i = str(i)
                method_res = {}
                # all_res[model_name][i][method_type]["result_summary"]["planned_cost"] = all_res[model_name][i][method_type]["all_res"][0]["planned_cost"]
                method_res[method_type] = {}
                
                json_file_path = os.path.join(result_dir, exp_name, method_type, 'experiment_results.json')
                with open(json_file_path, "r") as f:
                    all_res = json.load(f)
                model_name = list(all_res.keys())[0]

                method_res[method_type] = all_res[model_name][i][method_type]
                if method_type == 'CROWN':
                    inter_result.append(
                        [res["intermediate_best_outputs"] for res in method_res[method_type]["all_res"]]
                    )

                method_obj_dict[method_type].append(
                    method_res[method_type]["result_summary"][cost_type]
                )
        # inter_results[model_name] = inter_result
        stat = {method_type: method_obj_dict[method_type] for method_type in method_types}

        # json_file_path_new = os.path.join(result_dir, 'experiment_results_new.json')
        # with open(json_file_path_new, "w") as f:
        #     json.dump(all_res, f, indent=4, separators=(",", ": "))
        # print(f"data saved to {json_file_path_new}")
        vis_file_name_prefix = os.path.join(result_dir, exp_name)
        draw_box_plots(stat, method_types, task_name, True, label = "Planned Cost")
        output_file_name = os.path.join(vis_file_name_prefix, "planned_cost.svg")
        plt.savefig(output_file_name)
        plt.close()
        print(f"planned_cost saved to {output_file_name}")
        
        for method in method_types:
            if method != 'CROWN':
                plot_sampling_convergence(vis_file_name_prefix, method)

        plot_crown_convergence(vis_file_name_prefix, num_test)
        plot_crown_space(vis_file_name_prefix, num_test)

def draw_box_plots(
    planning_results, method_types, task_name, show_outlier, label
):
    """
    Draws box plots on the given axes for the specified cost type.

    Parameters:
    - planning_results: Data containing planning results.
    - method_types: List of method types.
    - colors: List of colors for the box plots.
    """
    colors = ["tomato", "skyblue", "lightgreen", "gold"]
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    data_for_plotting = []

    for method in method_types:
        data_for_plotting.append(planning_results[method])
    data_for_plotting = np.array(data_for_plotting).T

    font_size = 15
    width = 0.25

    bplot = ax.boxplot(
        data_for_plotting,
        positions=[1, 1.5, 2, 2.5],
        widths=width,
        patch_artist=True,
        # boxprops=dict(facecolor=colors),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker='d', color='black', markersize=4),
        showfliers=show_outlier,
    )
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    linewidth = 1
    for box in bplot['boxes']:
        box.set_linewidth(linewidth)
    for whisker in bplot['whiskers']:
        whisker.set_linewidth(linewidth)
    for cap in bplot['caps']:
        cap.set_linewidth(linewidth)
    for median in bplot['medians']:
        median.set_linewidth(linewidth)
    for outlier in bplot['fliers']:
        outlier.set_markeredgewidth(linewidth*1.5)

    ax.set_xticklabels(get_method_names_to_show(method_types), fontsize=font_size)
    # plt.xticks(rotation=30)
    # if task_name == 'Object Sorting' and label == 'Planned Cost':
    #     plt.ylim([4,18])
    #     plt.yticks(np.arange(4,17,3))
    ax.set_yticklabels(ax.get_yticks(), fontsize=font_size)
    # ax.set_title(f"{task_name}", fontsize=font_size)
    # if task_name != 'Object Sorting':
    #     num_yticks = 5 if task_name != 'Object Sorting' else 5
    #     ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks))
    ax.set_ylabel(label, fontsize=font_size, labelpad=5)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)

def get_method_names_to_show(method_types):
    method_list = []
    for method_type in method_types:
        if method_type == "GD_BF":
            method = "GD"
        elif method_type == "MPPI_BF":
            method = "MPPI"
        elif method_type == "DecentCEM":
            method = "CEM"
        elif method_type == "CROWN":
            method = "Ours"
        method_list.append(method)
    return method_list

def plot_crown_convergence(vis_file_name_prefix, num_test):
    json_file_path = os.path.join(vis_file_name_prefix, 'CROWN', 'experiment_results.json')
    with open(json_file_path, "r") as f:
        all_res = json.load(f)
    
    objective_list = []
    for i in range(num_test):
        objective_list.append(all_res[list(all_res.keys())[0]][str(i)]['CROWN']['all_res'][0]['intermediate_best_outputs'])

    objective_list = np.array(objective_list)
    color = '#2066a8'
    plot_convergence(objective_list, color, os.path.join(vis_file_name_prefix, 'CROWN_convergence.svg'))
    plot_convergence_case(objective_list, os.path.join(vis_file_name_prefix, 'CROWN_convergence_case.svg'))

def plot_sampling_convergence(vis_file_name_prefix, method):
    file_name = os.path.join(vis_file_name_prefix, method, f'{method}_full.txt')
    best_objectives = []

    with open(file_name, 'r') as file:
        content = file.readlines()
        case_count = sum(1 for line in content if "--------full--------" in line)
        pattern = r'restart (\d+): \[([^\]]+)\]'
        for line in content:
            match = re.search(pattern, line)
            if match:
                restart_index = int(match.group(1))  # Extract the restart index
                values = match.group(2)  # Extract the list of values as a string
                value_list = [float(v.strip()) for v in values.split(',')]  # Convert to a list of floats
                best_objectives.append(min(value_list))

    parsed_data = np.array(best_objectives).reshape(case_count, -1)
    # print(parsed_data.shape)
    # [num_cases, -1]
    parsed_data = np.minimum.accumulate(parsed_data, axis=1)
    color = '#ea801c'
    plot_convergence(parsed_data, color, os.path.join(vis_file_name_prefix, f'{method}_convergence.svg'))

def plot_crown_space(vis_file_name_prefix, num_test):
    json_file_path = os.path.join(vis_file_name_prefix, 'CROWN', 'experiment_results.json')
    with open(json_file_path, "r") as f:
        all_res = json.load(f)
    
    search_space_list = []
    total_space_list = []

    for i in range(num_test):
        space_size_dict = all_res[list(all_res.keys())[0]][str(i)]['CROWN']['all_res'][0]["space_size_dict"]
        search_space_list.append([1] + space_size_dict['search_space_size'])
        total_space_list.append([1] + space_size_dict['total_space_size'])
    
    search_space_list = np.array(search_space_list)
    total_space_list = np.array(total_space_list)
    color = '#2066a8'

    plot_convergence(search_space_list, color, os.path.join(vis_file_name_prefix, 'CROWN_search_space.svg'))
    plot_convergence(total_space_list, color, os.path.join(vis_file_name_prefix, 'CROWN_total_space.svg'))
    plot_convergence_case(search_space_list, os.path.join(vis_file_name_prefix, 'CROWN_search_space_case.svg'))
    plot_convergence_case(total_space_list, os.path.join(vis_file_name_prefix, 'CROWN_total_space_case.svg'))

def plot_convergence(data, color, plot_file):
    median = np.median(data, axis=0)
    percentile_25 = np.percentile(data, 25, axis=0)
    percentile_75 = np.percentile(data, 75, axis=0)
    plt.figure(figsize=(8, 6))
    plt.plot(median, color=color, linewidth=linewidth)
    plt.fill_between(range(data.shape[1]), percentile_25, percentile_75, color=color, alpha=alpha, linewidth=edge_width)
    # plt.ylim(ylim)
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    plt.close()

def plot_convergence_case(data, plot_file):
    # use rainbow color for different cases
    # colors = plt.cm.rainbow(np.linspace(0, 1, data.shape[0]))
    # viridis, crest, Blues, Spectral
    colors = sns.color_palette("coolwarm", data.shape[0])

    for i in range(data.shape[0]):
        plt.plot(data[i], color=colors[i], 
                #  linewidth=linewidth
                 )
    plt.xticks(np.arange(0, data.shape[1], 10))
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")

if __name__ == '__main__':
    linewidth = 3.5
    edge_width = 0
    alpha = 0.7

    # ylim = [29, 41]
    main()
