import os
import sys

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

sys.path.append(os.getcwd())
cwd = os.getcwd()
import traceback
import time
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import yaml
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    result_dir = 'result'
    # rope_2, pile_2, L_1, T_1
    exp_name_list = ['rope_1', 'pile_2', 'L_1', 'T_1']
    # exp_name_list = ['rope_2']
    for exp_name in exp_name_list:
        if 'rope' in exp_name:
            task_name = 'Rope Routing'
        elif 'pile' in exp_name:
            task_name = 'Object Sorting'
        elif 'L' in exp_name:
            task_name = 'Object Merging'
        elif 'T' in exp_name:
            task_name = 'Object Pushing w/ Obs.'
        
        json_file_path = os.path.join(result_dir, f'{exp_name}.json')
        with open(json_file_path, "r") as f:
            all_res = json.load(f)
        stat = {}
        inter_results = {}
        inter_result = []
        method_types = ['MPPI', 'MPPI_BF', 'Ours']
        model_name = list(all_res.keys())[0]
        num_test = 30
        cost_type = 'planned_cost'
        method_obj_dict = {method_type: [] for method_type in method_types}
        for i in range(num_test):
            i = str(i)
            method_res = {}
            for method_type in method_types:
                # all_res[model_name][i][method_type]["result_summary"]["planned_cost"] = all_res[model_name][i][method_type]["all_res"][0]["planned_cost"]
                if method_type == 'Ours':
                    method_type = 'CROWN'
                method_res[method_type] = {}
                method_res[method_type] = all_res[model_name][i][method_type]
                if method_type == 'CROWN':
                    inter_result.append(
                        [res["intermediate_best_outputs"] for res in method_res[method_type]["all_res"]]
                    )
            for method_type in method_types:
                method_obj_dict[method_type].append(
                    method_res[method_type if method_type != 'Ours' else 'CROWN']["result_summary"][cost_type]
                )
        # inter_results[model_name] = inter_result
        stat = {method_type: method_obj_dict[method_type] for method_type in method_types}

        # json_file_path_new = os.path.join(result_dir, 'experiment_results_new.json')
        # with open(json_file_path_new, "w") as f:
        #     json.dump(all_res, f, indent=4, separators=(",", ": "))
        # print(f"data saved to {json_file_path_new}")
        vis_file_name_prefix = os.path.join(result_dir, f'{exp_name}_')
        draw_box_plots(stat, method_types, task_name, False if 'pile' in exp_name else True, label = "Planned Cost")
        plt.savefig(vis_file_name_prefix + "planned_cost.svg")
        print(f"planned_cost saved to {vis_file_name_prefix + 'planned_cost.pdf'}")

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
    colors = ["tomato", "skyblue", "lightgreen"]
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    data_for_plotting = []

    for method in method_types:
        data_for_plotting.append(planning_results[method])
    data_for_plotting = np.array(data_for_plotting).T

    font_size = 15
    width = 0.25

    bplot = ax.boxplot(
        data_for_plotting,
        positions=[1.1, 1.8, 2.5],
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
    ax.set_xticklabels(method_types, fontsize=font_size)
    # plt.xticks(rotation=30)
    if task_name == 'Object Sorting' and label == 'Planned Cost':
        plt.ylim([4,18])
        plt.yticks(np.arange(4,17,3))
    ax.set_yticklabels(ax.get_yticks(), fontsize=font_size)
    # ax.set_title(f"{task_name}", fontsize=font_size)
    if task_name != 'Object Sorting':
        num_yticks = 5 if task_name != 'Object Sorting' else 5
        ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks))
    ax.set_ylabel(label, fontsize=font_size, labelpad=5)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)

if __name__ == '__main__':
    main()