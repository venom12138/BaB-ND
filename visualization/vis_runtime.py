import os
import numpy as np
import json
import matplotlib.pyplot as plt

file_dir = 't_model_size'
font_size = 38
def main():
    process_runtime()
    dict_file = os.path.join(file_dir, 'time_dict.json')
    with open(dict_file, 'r') as file:
        time_dict = json.load(file)
    # model_dict = {'169': [256,256], '170': [128,256,128], '176': [128,256,512,128],
    #                     '175': [128,256,256,256,128], '177': [256,256,256], 
    #                     '178': [128,256,256,128], '179': [128,256,512,512,256,128],}
    model_dict = {'169': [256,256], '176': [128,256,512,128],
                        '181': [128,256,512,256,128], '177': [256,256,256], 
                        '179': [128,256,512,512,256,128],}
    # parse model_size_dict to number of relu layer and total number of neurons
    model_size_dict = {model_id: (len(model_size), sum(model_size)) for model_id, model_size in model_dict.items()}
    print(model_size_dict)
    draw_cumulative_time(model_size_dict, time_dict)
    draw_runtime(model_size_dict, time_dict)

def draw_runtime(model_size_dict, time_dict):
    model_order = sorted(model_size_dict, key=lambda x: (model_size_dict[x][0], model_size_dict[x][1]))
    num_models = len(model_order)
    bounding_times = []
    branching_times = []
    searching_times = []
    total_times = []
    other_times = []

    for model_id in model_order:
        
        runtimes = time_dict[model_id]['runtimes']
        total_time = np.sum(runtimes['total_time'])
        bounding_time = np.sum(runtimes['bounding']) / total_time * 100
        searching_time = np.sum(runtimes['attack']) / total_time * 100
        branching_time = np.sum(runtimes['decision']) / total_time * 100
        other_time = (100 - bounding_time - searching_time - branching_time) 
        
        bounding_times.append(bounding_time)
        branching_times.append(branching_time)
        searching_times.append(searching_time)
        other_times.append(other_time)
        total_times.append(total_time)
    bar_width = 0.58
    model_labels = [f"{model_size_dict[model_id][1]}" for model_id in model_order]
    
    fig, ax = plt.subplots(figsize=(14, 10))

    # color_list = ["#f3cba6", "#ba8c78", "#90b89f", "#90b7bd"]
    color_list = ["#469393", "#e99945", "#469562", "#842421"]
    alpha = 0.8
    # add grid behind bars
    plt.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5, zorder=0)
    
    
    bar1 = np.array(bounding_times)
    bar2 = np.array(branching_times)
    bar3 = np.array(searching_times)
    bar4 = np.array(other_times)

    ax.barh(model_labels, bar1, color=color_list[0], edgecolor='grey', height=bar_width, label='Bounding', alpha=alpha)
    ax.barh(model_labels, bar2, left=bar1, color=color_list[1], edgecolor='grey', height=bar_width, label='Branching', alpha=alpha)
    ax.barh(model_labels, bar3, left=bar1+bar2, color=color_list[2], edgecolor='grey', height=bar_width, label='Searching', alpha=alpha)
    ax.barh(model_labels, bar4, left=bar1+bar2+bar3, color=color_list[3], edgecolor='grey', height=bar_width, label='Other', alpha=alpha)

    # ax.set_xlabel('Time (seconds)')
    # ax.set_title('Time Breakdown by Model')
    ax.legend(loc='lower center', bbox_to_anchor=(0.48, -0.45), ncol=2, fontsize=font_size, edgecolor='white', 
                        handletextpad=0.6, columnspacing=1.2)
    ax.set_xticks(np.arange(0, 101, 25))
    ax.set_xticklabels([f'{i}%' for i in np.arange(0, 101, 25)], fontsize=font_size)
    ax.set_yticklabels(model_labels, fontsize=font_size)
    ax.set_xlim(0, 100) 

    for label in ax.get_yticklabels():
        label.set_ha('center')
    ax.tick_params(axis='y', pad=55) 
    plt.subplots_adjust(left=0.13, right=0.93, top=0.91, bottom=0.3)
    runtime_file = os.path.join(file_dir, 'runtime.pdf')
    plt.savefig(runtime_file)
    print(f'Runtime plot saved to {runtime_file}')


def draw_cumulative_time(model_size_dict, time_dict):
    # reorder models with num_relu and num_neurons, if num_relu is the same, then sort by num_neurons
    model_order = sorted(model_size_dict, key=lambda x: (model_size_dict[x][0], model_size_dict[x][1]))
    num_models = len(model_order)
    print(model_order)
    color_list = ["#eec79f", "#f1dfa4", "#74b69f", "#a6cde4", "#e2c8d8"]
    # dark_color_list = ["#e99945", "#d6be6c", "#469562", "#75a59c", "#b88c99"]
    dark_color_list = ["#e99945", "#e1c064", "#579e7b", "#86b5d3", "#c29eb5"]
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    linewidth = 7.5
    for i, model_id in enumerate(model_order):
        model_size = model_size_dict[model_id]
        cumulative_times = np.array(time_dict[model_id]['cumulative_times'])
        mean_cumulative_time = np.mean(cumulative_times, axis=0)
        ax.plot(range(1, len(mean_cumulative_time) + 1), mean_cumulative_time, 
                label=f"{model_size[1]}" , color=color_list[i], linewidth=linewidth, alpha=1)
    
    linewidth = 1.5
    box_color = 'black'
    iterations = range(10, 51, 10)
    positions = np.array(iterations)
    data_for_boxplot = []
    for iter in iterations:
        data_for_boxplot.append([np.array(time_dict[model_id]['cumulative_times'])[:, iter-1] for model_id in model_order])
    data_for_boxplot = np.array(data_for_boxplot)
    box_width = 3
    
    for j in range(num_models):
        bplot = ax.boxplot(data_for_boxplot[:,j].T, positions=positions, 
                            widths=box_width, whis=[0, 100], patch_artist=True, boxprops=dict(edgecolor=box_color),
                            medianprops=dict(color=box_color), whiskerprops=dict(color=box_color), capprops=dict(color=box_color),)
        for patch in bplot['boxes']:
            patch.set_facecolor(dark_color_list[j])
        for box in bplot['boxes']:
            box.set_linewidth(linewidth)
        for whisker in bplot['whiskers']:
            whisker.set_linewidth(linewidth)
        for cap in bplot['caps']:
            cap.set_linewidth(linewidth)
        for median in bplot['medians']:
            median.set_linewidth(linewidth)

    plt.xlim(0,55)
    plt.ylim(0, 150)
    plt.xticks(range(10, 51, 10))
    plt.yticks(range(0, 151, 30))
    # plt.title('Mean Cumulative Time vs Iteration for Different Models')
    handles, labels = ax.get_legend_handles_labels()
    # Reverse the order
    handles = handles[::-1]
    labels = labels[::-1]
    # Create legend with reversed order
    ax.legend(handles, labels, title='# of ReLUs', markerscale=2, handlelength=4, handleheight = 4,
                handletextpad=2, labelspacing=1.5, borderaxespad=3)
    # set the font size of legend and label
    
    plt.setp(ax.get_legend().get_title(), fontsize=font_size)
    plt.setp(ax.get_legend().get_texts(), fontsize=font_size)

    plt.setp(ax.get_xticklabels(), fontsize=font_size)
    plt.setp(ax.get_yticklabels(), fontsize=font_size)
    plt.xlabel('Iterations', fontsize=font_size)
    plt.ylabel('Runtime (seconds)', fontsize=font_size)
    plt.subplots_adjust(left=0.13, right=0.93, top=0.91, bottom=0.15)
    ax.tick_params(axis='y', pad=10) 
    # plt.grid(True)
    cumulative_file = os.path.join(file_dir, 'cumulative_time.pdf')
    plt.savefig(cumulative_file)
    print(f'Cumulative time plot saved to {cumulative_file}')

def process_runtime():
    num_test = 5
    num_iter = 50
    num_record = num_test * num_iter
    file_path_dict = {file.split('_')[0]: os.path.join(file_dir, file) for file in os.listdir(file_dir) if file.endswith('.txt')}

    time_dict = {}
    for model_id, file_path in file_path_dict.items():
        end_times = []
        cumulative_times = []
        runtimes = {
            'total_time': [],
            'pickout': [],
            'decision': [],
            'bounding': [],
            'add_domain': [],
            'attack': []
        }

        # Open and read the file again
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'Result: unknown in' in line:
                    value = float(line.split('in')[1].split('seconds')[0].strip())
                    end_times.append(value)
                elif 'Cumulative time:' in line:
                    value = float(line.split('Cumulative time:')[1].strip())
                    cumulative_times.append(value)
                elif 'Total time:' in line:
                    parts = line.split()
                    runtimes['total_time'].append(float(parts[2]))
                    runtimes['pickout'].append(float(parts[4]))
                    runtimes['decision'].append(float(parts[6]))
                    runtimes['bounding'].append(float(parts[8]))
                    runtimes['add_domain'].append(float(parts[10]))
                    runtimes['attack'].append(float(parts[12]))

        assert len(end_times) == num_test
        assert len(cumulative_times) == num_record
        cumulative_times = np.array(cumulative_times).reshape(num_test, num_iter).tolist()
        for key, value in runtimes.items():
            assert len(value) == num_record
            value = np.array(value).reshape(num_test, num_iter).tolist()
        time_dict[model_id] = {
            'end_times': end_times,
            'cumulative_times': cumulative_times,
            'runtimes': runtimes
        }
    
    # save to json
    with open(os.path.join(file_dir, 'time_dict.json'), 'w') as file:
        json.dump(time_dict, file, indent=4)


if __name__ == '__main__':
    main()