import os
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import re

file_dir = '.'
mpl.rc('font', family='Arial')

def process_runtime():
    num_test = 1
    num_iter = 25
    num_record = num_test * num_iter
    file_path_dict = {file.split('.')[0]: os.path.join(file_dir, file) for file in os.listdir(file_dir) if file.endswith('.txt')}
    # file_path_dict = {"test": os.path.join(file_dir, "time_test.txt")}
    pattern = r"---------- (\w+): (\d+\.\d{8})----------"
    time_dict = {}
    for model_id, file_path in file_path_dict.items():
        end_times = []
        cumulative_times = []
        runtimes = {}

        with open(file_path, 'r') as file:
            contents = file.readlines()  # Read the file line by line

            # Loop over each line and find matches
            for line in contents:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1)
                    time = match.group(2)
                    if name in runtimes:
                        runtimes[name] += float(time)
                    else:
                        runtimes[name] = float(time)
                elif 'Result: unknown in' in line:
                    value = float(line.split('in')[1].split('seconds')[0].strip())
                    end_times.append(value)
                elif 'Cumulative time:' in line:
                    value = float(line.split('Cumulative time:')[1].strip())
                    cumulative_times.append(value)

        assert len(end_times) == num_test
        assert len(cumulative_times) == num_record
        cumulative_times = np.array(cumulative_times).reshape(num_test, num_iter).tolist()
        # for key, value in runtimes.items():
        #     assert len(value) == num_record
        #     value = np.array(value).reshape(num_test, num_iter).tolist()
        time_dict[model_id] = {
            'end_times': end_times,
            'cumulative_times': cumulative_times,
            'runtimes': runtimes
        }
    
    # save to json
    json_file = os.path.join(file_dir, 'time_dict.json')
    with open(json_file, 'w') as file:
        json.dump(time_dict, file, indent=4)

    print('Saved to', json_file)
    
def draw_runtime():
    with open('time_dict.json', 'r') as file:
        time_dict = json.load(file)
    model_list = [192, 384, 576, 768, 1152, 1536][::-1]
    bounding_times = []
    branching_times = []
    searching_times = []
    total_times = []
    other_times = []

    for model_id in model_list:
        model_id = str(model_id)
        runtimes = time_dict[model_id]['runtimes']
        total_time = time_dict[model_id]["end_times"][0]
        bounding_time = runtimes['get_nodes'] + runtimes['bound'] + runtimes['update_reference'] + runtimes['repeat_bound']
        searching_time = runtimes['searching']
        branching_time = runtimes['split'] + runtimes['pickout'] + runtimes['get_distribution']
        other_time = total_time - bounding_time - searching_time - branching_time
        
        bounding_times.append(bounding_time)
        branching_times.append(branching_time)
        searching_times.append(searching_time)
        other_times.append(other_time)
        total_times.append(total_time)
    bar_width = 0.58
    model_labels = calculate_mlp_params_vectorized(10, 8, model_list)
    
    fig, ax = plt.subplots(figsize=(7, 6))

    # color_list = ["#f3cba6", "#ba8c78", "#90b89f", "#90b7bd"]
    color_list = ["#469562", "#e99945", "#B22222", "#469393"]
    alpha = 0.8
    # add grid behind bars
    plt.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5, zorder=0)
    
    
    bar2 = np.array(bounding_times)
    bar3 = np.array(searching_times)
    bar1 = np.array(branching_times) + np.array(other_times)
    label_list = ['Branching & Other', 'Bounding', 'Searching']
    # bar4 = 

    ax.barh(model_labels, bar1, color=color_list[0], edgecolor='grey', height=bar_width, label=label_list[0], alpha=alpha)
    ax.barh(model_labels, bar2, left=bar1, color=color_list[1], edgecolor='grey', height=bar_width, label=label_list[1], alpha=alpha)
    ax.barh(model_labels, bar3, left=bar1+bar2, color=color_list[2], edgecolor='grey', height=bar_width, label=label_list[2], alpha=alpha)
    # ax.barh(model_labels, bar4, left=bar1+bar2+bar3, color=color_list[3], edgecolor='grey', height=bar_width, label='Other', alpha=alpha)

    label_size = 20
    tick_size = 20
    legend_size = 22
    # ax.set_xlabel('Runtime (seconds)', fontdict = {'fontsize': label_size})
    # ax.set_ylabel('Model size', fontdict = {'fontsize': label_size})
    # ax.set_title('Time Breakdown by Model')
    # ax.legend(loc='lower center', bbox_to_anchor=(0.48, -0.45), ncol=2, fontsize=legend_size, edgecolor='white', 
    #                     handletextpad=0.6, columnspacing=1.2)
    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend([handles[0]], [labels[0]], loc='upper left', bbox_to_anchor=(-0.05, -0.15), fontsize=legend_size, edgecolor='white', handletextpad=0.6, columnspacing=1.2, ncol=1)
    legend2 = ax.legend(handles[1:], labels[1:], loc='upper left', bbox_to_anchor=(-0.05, -0.25), fontsize=legend_size, edgecolor='white', handletextpad=0.6, columnspacing=1.2, ncol=2)

    ax.add_artist(legend1)
    ax.set_xticks(np.arange(0, 101, 25))
    ax.set_xticklabels([f'{i}' for i in np.arange(0, 101, 25)], fontsize=tick_size)
    ax.set_yticklabels(model_labels, fontsize=tick_size)
    # ax.set_xlim(0, 100) 

    for label in ax.get_yticklabels():
        label.set_ha('center')
    ax.tick_params(axis='y', pad=40) 
    plt.subplots_adjust(left=0.13, right=0.93, top=0.91, bottom=0.3)
    plt.tight_layout()
    runtime_file = os.path.join(file_dir, 'runtime.pdf')
    plt.savefig(runtime_file)
    print(f'Runtime plot saved to {runtime_file}')

def calculate_mlp_params_vectorized(input_dim, output_dim, num_ReLU_list):
    # Convert num_ReLU_list to a numpy array if it's not already
    num_ReLU_list = np.array(num_ReLU_list)
    
    # Define the proportions for each layer
    layer_1 = num_ReLU_list // 6
    layer_2 = num_ReLU_list // 3
    layer_3 = num_ReLU_list // 3
    layer_4 = num_ReLU_list // 6

    # Calculate parameters for each layer
    params_layer_1 = input_dim * layer_1 + layer_1
    params_layer_2 = layer_1 * layer_2 + layer_2
    params_layer_3 = layer_2 * layer_3 + layer_3
    params_layer_4 = layer_3 * layer_4 + layer_4
    params_output_layer = layer_4 * output_dim + output_dim

    # Total number of parameters for each num_ReLU value
    total_params = (params_layer_1 + params_layer_2 + 
                    params_layer_3 + params_layer_4 + params_output_layer)
    total_params_in_k = total_params / 1000  # Convert to thousands
    formatted_params = np.round(total_params_in_k, 4)  # Round to 4 significant digits
    formatted_params_str = [f"{p:.3f}K" if p < 10 else f"{p:.2f}K" if p < 100 else f"{p:.1f}K" for p in formatted_params]
    return formatted_params_str


if __name__ == '__main__':
    process_runtime()
    draw_runtime()