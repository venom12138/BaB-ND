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
from vis_planning import draw_box_plots


def main():
    result_dir = 'result'
    # rope_2, pile_1, L_1, T_1
    exp_name_list = ['rope_2', 'pile_1', 'L_1', 'T_1']
    # exp_name_list = ['L_1', 'T_1']
    # exp_name_list = ['T_1']
    method_types = ['MPPI', 'MPPI_BF', 'Ours']
    for exp_name in exp_name_list:
        if 'rope' in exp_name:
            task_name = 'Rope Routing'
            stat = {
                "Ours":     [1, 0, 0, 1, 1, 0, 1, 1, 1, 1],
                "MPPI_BF":  [1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                "MPPI":     [1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
            }
        elif 'pile' in exp_name:
            task_name = 'Object Sorting'
            stat = {
                "Ours": [3.945, 4.212, 3.293, 4.529, 2.3591],
                "MPPI_BF": [5.754, 3.414, 4.023, 5.3097, 5.1264],
                "MPPI": [5.6491, 6.4122, 4.2242, 5.3337, 7.4176]
            }
        elif 'L' in exp_name:
            task_name = 'Object Merging'
            stat = {
                "Ours":     [1.3158, 0.6867, 0.3724, 1.1184, 1.5320, 1.1296, 3.7443],
                "MPPI_BF":  [1.0482, 3.6798, 2.2508, 1.3710, 2.5742, 1.0987, 2.1740],
                "MPPI":     [3.8059, 2.6877, 3.9124, 1.4697, 2.8779, 4.8434, 4.8758]
            }
        elif 'T' in exp_name:
            task_name = 'Object Pushing w. Obs.'
            stat = {
                "Ours":     [0.54029, 0.65753, 0.26819, 0.16197, 1.38505, 1.73695, 0.6161, 0.78291, 0.77838, 4.80185],
                "MPPI_BF":  [1.2534, 1.4318, 6.1467, 0.8247, 1.6616, 0.7004, 1.4955, 5.6506, 1.31, 0.9809],
                "MPPI":     [3.1298, 1.8925, 3.4412, 5.1853, 3.2427, 2.713, 5.0463, 1.9194, 5.3043, 6.0558]
            }
        vis_file_name_prefix = os.path.join(result_dir, f'{exp_name}_')
        if 'rope' in exp_name:
            draw_success_rate(stat, method_types, task_name, label="Success Rate")
            plt.savefig(vis_file_name_prefix + "success_rate.svg")
            print(f"success_rate saved to {vis_file_name_prefix + 'success_rate.pdf'}")
        else:
            draw_box_plots(stat, method_types, task_name, True, label="Final Cost")
            plt.savefig(vis_file_name_prefix + "final_cost.svg")
            print(f"final_cost saved to {vis_file_name_prefix + 'final_cost.pdf'}")

def draw_success_rate(stat, method_types, task_name, label="Success Rate"):
    colors = ["tomato", "skyblue", "lightgreen"]
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    x = [1.1, 1.8, 2.5]
    font_size = 15
    width = 0.25
    success_rate = [sum(stat[method_type]) / len(stat[method_type]) for method_type in method_types]
    for i, method_type in enumerate(method_types):
        ax.bar(x[i], success_rate[i], width=width, label=method_type, color=colors[i], edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(method_types)
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0], font_size = font_size)
    ax.set_yticklabels(ax.get_yticks(), fontsize=font_size)
    # ax.set_title(f"{task_name}", fontsize=font_size)
    ax.set_ylabel(label, fontsize=font_size, labelpad=5)
    ax.set_xticklabels(method_types, fontsize=font_size)
    ax.set_xlim(0.6, 3)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
    # fig.tight_layout()

if __name__ == '__main__':
    main()