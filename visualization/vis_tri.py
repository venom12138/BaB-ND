import os
import sys

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

sys.path.append(os.getcwd())
cwd = os.getcwd()
import numpy as np
import json
import matplotlib.pyplot as plt
import re

def convert_data_crown(method):
    json_file_path = f'exp/pushing_T/0405_pushing_T/H0153_use_augdata,online_training=0,sample_epoch=1,bs=1024,epochs=5/09_10_planning_pushing_T_id2/crown_{method}/experiment_results.json'
    with open(json_file_path, "r") as f:
        all_res = json.load(f)
    
    objective_list = []
    for i in range(10):
        objective_list.append(all_res['768'][str(i)]['CROWN']['all_res'][0]['intermediate_best_outputs'])

    objective_list = np.array(objective_list)

    np.save(os.path.join(data_dir, f"crown_{method}.npy"), objective_list)

def convert_data_sampling(method):
    if method == 'cem':
        file_name = "exp/pushing_T/0405_pushing_T/H0153_use_augdata,online_training=0,sample_epoch=1,bs=1024,epochs=5/09_10_planning_pushing_T_id2/CEM/decentCEM.txt"
    else:
        file_name = "exp/pushing_T/0405_pushing_T/H0153_use_augdata,online_training=0,sample_epoch=1,bs=1024,epochs=5/09_10_planning_pushing_T_id2/MPPI/mppi_bf.txt"
    best_objectives = []

    with open(file_name, 'r') as file:
        content = file.readlines()
        case_count = sum(1 for line in content if "--------summary--------" in line)
        for line in content:
            match = re.search(r'n_restarts:\d+, best cost: (\d+\.\d+)', line)
            if match:
                best_objectives.append(float(match.group(1)))
    parsed_data = np.array(best_objectives).reshape(case_count, -1)
    # [num_cases, -1]

    np.save(os.path.join(data_dir, f"{method}_restart.npy"), parsed_data)

def convert_data_sampling_full(method):
    if method == 'cem':
        file_name = "exp/pushing_T/0405_pushing_T/H0153_use_augdata,online_training=0,sample_epoch=1,bs=1024,epochs=5/09_10_planning_pushing_T_id2/CEM/decentCEM_full.txt"
    else:
        file_name = "exp/pushing_T/0405_pushing_T/H0153_use_augdata,online_training=0,sample_epoch=1,bs=1024,epochs=5/09_10_planning_pushing_T_id2/MPPI/mppi_bf_full.txt"
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
    print(parsed_data.shape)
    # [num_cases, -1]
    parsed_data = np.minimum.accumulate(parsed_data, axis=1)

    np.save(os.path.join(data_dir, f"{method}_restart_full.npy"), parsed_data)

def plot_crown(method):
    color = '#2066a8'
    crown_data = np.load(os.path.join(data_dir, f"crown_{method}.npy"))
    n_cases, n_iteration = crown_data.shape
    median = np.median(crown_data, axis=0)
    percentile_25 = np.percentile(crown_data, 25, axis=0)
    percentile_75 = np.percentile(crown_data, 75, axis=0)
    plt.figure(figsize=(8, 6))
    plt.plot(median, color=color, linewidth=linewidth)
    plt.fill_between(range(n_iteration), percentile_25, percentile_75, color=color, alpha=alpha, linewidth=edge_width)
    plt.ylim(ylim)
    # plt.xticks(fontsize=font_size)
    # plt.yticks(fontsize=font_size)
    # plt.title('Convergence Curve')
    # plt.xlabel('Iteration')
    # plt.ylabel('Optimized Objective')
    # plt.legend()
    # plt.grid(True)
    plot_file = os.path.join(vis_dir, f'crown_{method}.svg')
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

def plot_sampling(method):
    sampling_data = np.load(os.path.join(data_dir, f"{method}_restart.npy"))
    n_cases, n_report = sampling_data.shape
    report_interval = 10
    
    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.boxplot(sampling_data)
    plt.ylim(ylim)
    # plt.title('Optimized Objective per 10 Agents')
    # plt.xlabel('Agent (every 10th)')
    # plt.ylabel('Optimized Objective')
    # plt.xticks(ticks=np.arange(1, n_report + 1), labels=np.arange(1, n_report + 1) * report_interval)
    # plt.legend()
    # plt.grid(True)
    plot_file = os.path.join(vis_dir, f'{method}_restart.svg')
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

def plot_sampling_custom(method):
    sampling_data = np.load(os.path.join(data_dir, f"{method}_restart.npy"))
    n_cases, n_report = sampling_data.shape
    report_interval = 10
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_report):
        custom_boxplot(ax, sampling_data[:, i], position=i+1, whisker_multiplier=1., box_width=0.5, box_color='blue', median_color='red', outlier_color='red')

    # plt.title('Optimized Objective per 10 Agents')
    # plt.xlabel('Agent (every 10th)')
    # plt.ylabel('Optimized Objective')
    # plt.xticks(ticks=np.arange(1, n_report + 1), labels=np.arange(1, n_report + 1) * report_interval)
    # plt.legend()
    # plt.grid(True)
    plot_file = os.path.join(vis_dir, f'{method}_restart_custom.svg')
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    

def custom_boxplot(ax, data, position=1, whisker_multiplier=1.5, box_width=0.5, box_color='blue', median_color='red', outlier_color='red'):
    """
    Draws a custom box plot on the provided Axes object.

    Parameters:
    - ax: The matplotlib Axes object to draw the box plot on.
    - data: The data array to plot.
    - position: The position on the x-axis to center the box plot (default is 1).
    - whisker_multiplier: The multiplier for the interquartile range (default is 1.5).
    - box_width: The width of the box (default is 0.5).
    - box_color: Color of the box (default is 'blue').
    - median_color: Color of the median line (default is 'red').
    - outlier_color: Color of the outliers (default is 'red').
    """
    
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    whisker_width = whisker_multiplier * iqr

    lower_whisker = q1 - whisker_width
    upper_whisker = q3 + whisker_width
    
    outliers = data[(data < lower_whisker) | (data > upper_whisker)]
    normal_data = data[(data >= lower_whisker) & (data <= upper_whisker)]

    ax.add_patch(plt.Rectangle((position - box_width/2, q1), box_width, q3-q1, fill=True, color=box_color, alpha=0.5))
    ax.plot([position - box_width/2, position + box_width/2], [median, median], color=median_color)
    ax.plot([position, position], [q1, normal_data.min()], color='black')  # Lower whisker
    ax.plot([position, position], [q3, normal_data.max()], color='black')  # Upper whisker
    ax.plot(np.full_like(outliers, position), outliers, 'o', color=outlier_color)

    return ax

def plot_sampling_full(method):
    color = '#ea801c'
    sampling_data = np.load(os.path.join(data_dir, f"{method}_restart_full.npy"))
    n_cases, n_restart = sampling_data.shape
    report_interval = 10
    median = np.median(sampling_data, axis=0)
    percentile_25 = np.percentile(sampling_data, 25, axis=0)
    percentile_75 = np.percentile(sampling_data, 75, axis=0)
    plt.figure(figsize=(8, 6))
    plt.plot(median, color=color, linewidth=linewidth)
    plt.fill_between(range(n_restart), percentile_25, percentile_75, color=color, alpha=alpha, linewidth=edge_width)
    plt.ylim(ylim)

    # plt.xticks(fontsize=font_size)
    # plt.yticks(fontsize=font_size)

    # plt.title('Optimized Objective per 10 Agents')
    # plt.xlabel('Agent (every 10th)')
    # plt.ylabel('Optimized Objective')
    # plt.xticks(ticks=np.arange(1, (n_restart + 1) // report_interval, 1), labels=np.arange(1, n_restart + 1, report_interval))
    # plt.legend()
    # plt.grid(True)
    plot_file = os.path.join(vis_dir, f'{method}_restart_full.svg')
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

def parse_closed_loop_objectives():
    file_name = os.path.join(data_dir, "close_loop_cost.txt")
    methods_data = {}
    current_method = None

    with open(file_name, 'r') as file:
        content = file.readlines()

        for line in content:
            line = line.strip()  # Remove any leading/trailing spaces
            if line.endswith(":"):  # This indicates a method name
                current_method = line[:-1]  # Remove the colon and set the current method
                methods_data[current_method] = []  # Initialize an empty list for this method
            elif current_method and line:  # If it's not a method name and not empty
                # Convert the objectives to floats and add to the current method's list
                objectives = [float(obj.strip()) for obj in line.split(',')]
                methods_data[current_method].extend(objectives)

    return methods_data


def plot_closed_loop_objectives():
    methods_data = parse_closed_loop_objectives()
    width = 0.3
    colors =["#ea801c", "#2066a8"]
    ylim_box = [0.01, 0.99]

    for base in ["CEM", "MPPI"]:
        methods_to_plot = [base, f"CROWN_{base}"]
        selected_data = [methods_data[method] for method in methods_to_plot]
        fig, ax = plt.subplots(figsize=(5, 5))
        draw_box_plot(ax, selected_data, False, colors, width, ylim_box)
        file_name = os.path.join(vis_dir, f"{base}_close_loop.svg")
        plt.savefig(file_name)
        plt.close()
        print(f"Plot saved to {file_name}")


def plot_closed_loop_objectives_new():
    methods_data = parse_closed_loop_objectives()
    width = 0.3
    colors =["#ea801c", "#2066a8"]
    ylim_normal = [0.01, 0.9]
    ylim_outlier = [2, 3]
    height_ratios = np.array([1, 3])
    height_ratios = height_ratios / height_ratios.min()

    for base in ["CEM", "MPPI"]:
        methods_to_plot = [base, f"CROWN_{base}"]
        selected_data = [methods_data[method] for method in methods_to_plot]

        outliers = [value for data in selected_data for value in data if value > 1]
        if len(outliers) > 0:
            ylim_scale = 0.1
            ylim_outlier = [min(outliers) - ylim_scale, max(outliers) + ylim_scale]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5.1, 5), gridspec_kw={'height_ratios': height_ratios})
        # ax1: the top plot, ax2: the bottom plot
        draw_box_plot(ax1, selected_data, True, colors, width, ylim_outlier)
        draw_box_plot(ax2, selected_data, True, colors, width, ylim_normal)
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)
        ax2.xaxis.tick_bottom()
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        diagonal_height = d * height_ratios[1]
        ax1.plot((-d, +d), (-diagonal_height, +diagonal_height), **kwargs)        # top-left diagonal
        ax1.plot((1  - d, 1 + d), (-diagonal_height, +diagonal_height), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        file_name = os.path.join(vis_dir, f"{base}_close_loop_new.svg")
        plt.savefig(file_name)
        plt.close()
        print(f"Plot saved to {file_name}")


def draw_box_plot(ax: plt.Axes, selected_data, showfliers, colors, width, ylim_box):
    bplot = ax.boxplot(selected_data,
        positions=[0.6, 1.4],
        widths=width,
        patch_artist=True,
        # boxprops=dict(facecolor=colors),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker='d', color='black', markersize=4),
        showfliers=showfliers,
        
    )
    ax.set_xticks([])
    ax.set_ylim(ylim_box)
    ytick_fontsize = 18
    # set font size for the axis ticks
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ytick_fontsize)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    linewidth = 1.5
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


if __name__ == '__main__':
    save_dir = "./result_tri"
    os.makedirs(save_dir, exist_ok=True)
    data_dir = os.path.join(save_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # font_family = 'Arial'
    font_size = 22
    plt.rcParams['font.size'] = font_size
    linewidth = 3.5
    edge_width = 0
    alpha = 0.7

    ylim = [29, 41]

    # for method in ['cem', 'mppi']:
    #     convert_data_crown(method)
    #     plot_crown(method)
    #     convert_data_sampling_full(method)
    #     plot_sampling_full(method)

    # parse_closed_loop_objectives()
    plot_closed_loop_objectives_new()

