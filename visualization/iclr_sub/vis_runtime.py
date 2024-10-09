import os, sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib
import matplotlib as mpl
mpl.rc('font', family='Arial')

def main():
    file_dirs = ['MIP', 'CROWN']
    method_runtime_dict = {}
    for file_dir in file_dirs:
        method_runtime_dict[file_dir] = fetch_runtime(file_dir)
    draw_heatmap3(method_runtime_dict["MIP"], method_runtime_dict["CROWN"])

def draw_heatmap(runtime_dict1, runtime_dict2):
    # Extract unique model sizes and horizons for both dicts
    all_model_sizes = sorted(set([key[0] for key in runtime_dict1.keys()] + [key[0] for key in runtime_dict2.keys()]))
    all_horizons = sorted(set([key[1] for key in runtime_dict1.keys()] + [key[1] for key in runtime_dict2.keys()]))

    # Create two 2D arrays to store runtime values
    heatmap_data1 = np.zeros((len(all_horizons), len(all_model_sizes)))
    heatmap_data2 = np.zeros((len(all_horizons), len(all_model_sizes)))

    # Fill the arrays with runtime values from the dictionaries
    for (model_size, horizon), runtime in runtime_dict1.items():
        row = all_horizons.index(horizon)
        col = all_model_sizes.index(model_size)
        if runtime == 'Fail':
            heatmap_data1[row, col] = 650
        elif runtime == 'Subopt':
            heatmap_data1[row, col] = 600
        else:
            heatmap_data1[row, col] = runtime

    for (model_size, horizon), runtime in runtime_dict2.items():
        row = all_horizons.index(horizon)
        col = all_model_sizes.index(model_size)
        heatmap_data2[row, col] = runtime

    # Create subplots for side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    # Normalize the color map between both heatmaps
    vmin = min(np.min(heatmap_data1), np.min(heatmap_data2))
    vmax = max(np.max(heatmap_data1), np.max(heatmap_data2))

    # Plot first heatmap
    im1 = axes[0].imshow(heatmap_data1, cmap='Spectral_r', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_xticks(np.arange(len(all_model_sizes)))
    axes[0].set_xticklabels(all_model_sizes)
    axes[0].set_yticks(np.arange(len(all_horizons)))
    axes[0].set_yticklabels(all_horizons)
    axes[0].set_xlabel('Model Size')
    axes[0].set_ylabel('Horizon')
    axes[0].set_title('Heatmap 1')

    # Plot second heatmap
    im2 = axes[1].imshow(heatmap_data2, cmap='Spectral_r', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_xticks(np.arange(len(all_model_sizes)))
    axes[1].set_xticklabels(all_model_sizes)
    axes[1].set_xlabel('Model Size')
    axes[1].set_title('Heatmap 2')

    # Create a common color bar for both subplots
    cbar = fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Runtime (seconds)')

    # plt.tight_layout()
    plt.savefig('runtime_heatmaps.pdf')
    plt.close()

def draw_heatmap2(runtime_dict1, runtime_dict2):
    df_status_1 = dict_to_dataframe(runtime_dict1)
    df_status_2 = dict_to_dataframe(runtime_dict2)
    df_numeric_1 = df_status_1.replace({'Subopt': 600, 'Fail': 800})
    df_numeric_2 = df_status_2.replace({'Subopt': 600, 'Fail': 800})

    stacked_1 = df_numeric_1.stack()
    stacked_2 = df_numeric_2.stack()
    stacked_2.index = pd.MultiIndex.from_tuples([(h, f"{m}_2") for h, m in stacked_2.index])

    concatenated = pd.concat([stacked_1, stacked_2])
    ranked_concatenated = concatenated.rank(method='dense')
    ranked_values_1 = ranked_concatenated.loc[stacked_1.index].unstack()
    ranked_values_2 = ranked_concatenated.loc[stacked_2.index].unstack()
    max_rank = ranked_concatenated.max()
    ranked_values_1 = ranked_values_1 / max_rank
    ranked_values_2 = ranked_values_2 / max_rank

    ranked_values_1[ranked_values_1 <= 0.8] *= 0.8
    ranked_values_2[ranked_values_2 <= 0.8] *= 0.8

    ranked_values_1.replace({max_rank-1: 0.9, max_rank: 1.0}, inplace=True)
    ranked_values_2.replace({max_rank-1: 0.9, max_rank: 1.0}, inplace=True)

    # Use 'Spectral_r' diverging palette
    palette = sns.color_palette("Spectral_r", as_cmap=True, n_colors=31)

    # Create the heatmap plot with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot the first heatmap
    sns.heatmap(ranked_values_1, annot=df_status_1, cmap=palette, cbar=False,
                ax=axes[0], linewidths=0.5, linecolor='white', annot_kws=dict(size=14), fmt='')

    # Plot the second heatmap
    sns.heatmap(ranked_values_2, annot=df_status_2, cmap=palette, cbar=False,
                ax=axes[1], linewidths=0.5, linecolor='white', annot_kws=dict(size=14), fmt='')

    # Set labels and titles
    axes[0].set_title('Heatmap 1')
    axes[1].set_title('Heatmap 2')
    axes[0].set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
    axes[1].set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)

    # Add a shared color bar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

    plt.tight_layout()
    plt.savefig('heatmaps_shared_ranking.pdf')



def fill_heatmap_data(runtime_dict, all_horizons, all_model_sizes):
    """Helper function to fill the heatmap data array."""
    heatmap_data = np.zeros((len(all_horizons), len(all_model_sizes)))

    for (model_size, horizon), runtime in runtime_dict.items():
        row = all_horizons.index(horizon)
        col = all_model_sizes.index(model_size)
        if runtime == 'Fail':
            heatmap_data[row, col] = 350  # Arbitrary high value for 'Fail'
        elif runtime == 'Subopt' or runtime >= 300:
            heatmap_data[row, col] = 300  # Arbitrary value for 'Subopt'
        else:
            heatmap_data[row, col] = runtime
    return heatmap_data

def draw_single_heatmap(heatmap_data, all_model_sizes, all_horizons, vmin, vmax, output_file):
    """Helper function to draw a single heatmap with consistent formatting."""
    # crest, Spectral_r
    fig, ax = plt.subplots(figsize=(6, 6))
    formatted_data = np.vectorize(format_annotation)(heatmap_data)
    sns.heatmap(heatmap_data, ax=ax, cmap="Spectral_r", 
                vmin=vmin, vmax=vmax, fmt='',
                annot=formatted_data,
                linewidths=0.5, linecolor='white',
                cbar=False, xticklabels=calculate_mlp_params_vectorized(10, 8, all_model_sizes),
                yticklabels=all_horizons,
                annot_kws=dict(size=fontsize))

    # ax.set_title(title)
    # lable_size = fontsize + 3
    # ax.set_xlabel('Model Size', fontdict = {"fontsize": lable_size})
    # ax.set_ylabel('Horizon', fontdict = {"fontsize": lable_size})
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f'Heatmap saved to {output_file}')
    plt.close()
    

def format_annotation(val):
    if val >= 350:
        return 'Fail'
    elif val >= 100:
        return f"{val:.1f}" 
    elif val >= 10:
        return f"{val:.2f}"  # Two decimals if the number is 10 or greater
    # elif val >= 1:
    #     return f"{val:.3f}"  # Three decimals if the number is between 1 and 10
    else:
        return f"{val:.3f}"

def draw_heatmap3(runtime_dict1, runtime_dict2):
    # Extract unique model sizes and horizons for both dicts
    all_model_sizes = sorted(set([key[0] for key in runtime_dict1.keys()] + [key[0] for key in runtime_dict2.keys()]))
    all_horizons = sorted(set([key[1] for key in runtime_dict1.keys()] + [key[1] for key in runtime_dict2.keys()]))

    # Fill the heatmap data
    heatmap_data1 = fill_heatmap_data(runtime_dict1, all_horizons, all_model_sizes)
    heatmap_data2 = fill_heatmap_data(runtime_dict2, all_horizons, all_model_sizes)

    # Normalize the color map between both heatmaps
    vmin = 0
    vmax = 350

    # Plot heatmaps using helper function
    draw_single_heatmap(heatmap_data1, all_model_sizes, all_horizons, vmin, vmax, 'runtime_heatmap1.pdf')
    draw_single_heatmap(heatmap_data2, all_model_sizes, all_horizons, vmin, vmax, 'runtime_heatmap2.pdf')
    draw_colorbar(vmin, vmax, 'runtime_colorbar.pdf')

def draw_colorbar(vmin, vmax, output_file):
    # fig, ax = plt.subplots(figsize=(6, 6))
    # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='Spectral_r'), 
    #                     ax=ax, orientation='vertical', 
    #                     # fraction=0.02, 
    #                     pad=0.04
    #                     )
    # cbar.set_ticks([50, 100, 150, 200, 250, 300, 350])
    # cbar.set_ticklabels(['50', '100', '150', '200', '250', '300', 'Fail'], {"fontsize": fontsize})

    # cbar.set_label('Runtime (seconds)', fontdict = {"fontsize": fontsize})
    # plt.savefig(output_file)
    # print(f'Colorbar saved to {output_file}')
    # plt.close()
    fig, ax = plt.subplots(figsize=(1.5, 6))  # Smaller and narrower figure
    plt.axis('off')  # Turn off axis
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.cm.Spectral_r

    # Create a ScalarMappable with the norm and cmap to construct the colorbar
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Positioning the colorbar
    cax = fig.add_axes([0.05, 0.05, 0.25, 0.9])  # Adjust the dimensions as necessary

    # Create the colorbar
    cbar = fig.colorbar(mappable, cax=cax, orientation='vertical')
    # cbar.set_ticks([0, 50, 100, 150, 200, 250, 300, 350])
    # cbar.set_ticklabels(['0', '50', '100', '150', '200', '250', '300', 'Fail'])
    cbar.set_ticks([0, 100, 200, 300, 350])
    cbar.set_ticklabels(['0', '100', '200', '300', 'Fail'])
    cbar.ax.tick_params(labelsize=fontsize+4)

    # cbar.set_label('Runtime (seconds)', fontsize=fontsize)

    # Saving the file
    plt.tight_layout()
    plt.savefig(output_file)
    print(f'Colorbar saved to {output_file}')
    plt.close(fig)  # Close the figure to free up memory


def dict_to_dataframe(input_dict):
    model_sizes = sorted(set([key[0] for key in input_dict.keys()]))
    horizons = sorted(set([key[1] for key in input_dict.keys()]))
    df = pd.DataFrame(index=horizons, columns=model_sizes)

    for (model_size, horizon), value in input_dict.items():
        df.at[horizon, model_size] = value

    return df


def fetch_runtime(file_dir):

    sub_dirs = os.listdir(file_dir)
    
    model_size_list = [24, 48, 96, 384, 768, 1536]
    horion_list = [1, 3, 5, 10, 15, 20]
    runtime_dict = {}
    for model_size in model_size_list:
        for horizon in horion_list:
            runtime_dict[(model_size, horizon)] = 'Fail'

    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(file_dir, sub_dir)
        try:
            method = sub_dir.split('_')[-3].upper()
            model_size = int(sub_dir.split('_')[-2])
            horizon = int(sub_dir.split('_')[-1])
        except:
            continue
        json_file_name = os.path.join(sub_dir_path, 'experiment_results.json')
        if not os.path.exists(json_file_name):
            continue
        with open(json_file_name, 'r') as f:
            data = json.load(f)
        
        test_cases = data[list(data.keys())[0]]
        avg_runtime = np.mean([test_cases[test_id][method]["result_summary"]["runtime"] for test_id in test_cases])
        if avg_runtime >= 300:
            avg_runtime = 'Subopt'
        runtime_dict[(model_size, horizon)] = avg_runtime
    return runtime_dict


def merge_json_files(file1, file2, output_file):
    with open(file1, 'r') as f1:
        data1 = json.load(f1)
    with open(file2, 'r') as f2:
        data2 = json.load(f2)
    
    assert len(data1.keys()) == len(data2.keys()) == 1
    assert list(data1.keys())[0] == list(data2.keys())[0]

    model_name = list(data1.keys())[0]

    max_key1 = max(map(int, data1[model_name].keys())) if data1 else -1
    new_data2 = {}
    for i, (key, value) in enumerate(data2[model_name].items()):
        new_data2[str(max_key1 + 1 + i)] = value

    merged_data = {**data1[model_name], **new_data2}
    merged_data = {model_name: merged_data}
    with open(output_file, 'w') as out_f:
        json.dump(merged_data, out_f, indent=4)

    print(f'Files merged into {output_file}')

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
    fontsize = 18
    main()
    # fetch_runtime()
    # merge_json_files('mip/mip_24_10_3/experiment_results.json', 'mip/mip_24_10_2/experiment_results.json', 'mip/mip_24_10/experiment_results.json')
