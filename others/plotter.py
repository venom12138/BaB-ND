import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import math
import os
from PIL import Image
from scipy.ndimage import median_filter

from tasks.box_pushing import box_sim
from tasks.pushing_T import t_sim
from tasks.merging_L import l_sim
from tasks.inserting import hp_sim
from others.helper import *

def plot_search_space_size(space_size_list, file_name, start_iter=0):
    # space_size_list: list of list [num_cases, num_iters]
    space_size_list = np.array(space_size_list)
    num_cases, num_iters = space_size_list.shape
    space_size_list = space_size_list[:, start_iter:]
    iteration_indices = np.arange(start_iter+1, num_iters+1)
    xticks = np.arange(start_iter+1, num_iters+1, max((num_iters+1)//5, 1))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    median_color, percentile_color = '#197ab7', '#a5cfe3'
    median = np.median(space_size_list, axis=0)
    percentile_25 = np.percentile(space_size_list, 25, axis=0)
    percentile_75 = np.percentile(space_size_list, 75, axis=0)
    axes[0].plot(iteration_indices, median, color=median_color, label='Median')
    axes[0].fill_between(iteration_indices, percentile_25, percentile_75, color=percentile_color, edgecolor=median_color, alpha=0.5, label='25th-75th Percentile')
    axes[0].set_xticks(xticks)
    axes[0].set_xlim([start_iter, num_iters])
    axes[0].set_ylim([0, 1])
    # plt.set_cmap(cm.get_cmap('Spectral', num_cases))
    for i in range(num_cases):
        axes[1].plot(iteration_indices, space_size_list[i], alpha=1)
    axes[1].set_xticks(xticks)
    axes[1].set_xlim([start_iter, num_iters])
    axes[1].set_ylim([0, 1])
    plt.savefig(file_name)
    print(f"Saved to {file_name}")
    pass


def convert_and_denoise_gif(input_filename, color_mapping):
    """
    Convert the colors of a GIF image based on a provided color mapping and denoise it.
    The new image is saved with '_converted' appended to the original file name before the extension.

    Parameters:
    - input_filename (str): The file path for the input GIF image.
    - color_mapping (dict): A dictionary where keys are tuples representing the original colors
                            and values are tuples representing the new colors.
    # color_map = {
    #     (0, 0, 0, 255): (255, 255, 255, 255),       # Black to white
    #     (0, 0, 255, 255): (128, 0, 128, 255),       # Blue to purple
    #     (255, 255, 0, 255): (0, 128, 0, 255)        # Yellow to green
    # }
    """
    try:
        # Load the input image
        img = Image.open(input_filename)
        img_array = np.array(img)

        # Apply the color mapping to each pixel including the alpha channel
        for original_color, new_color in color_mapping.items():
            mask = np.all(img_array[:, :, :4] == original_color, axis=-1)
            img_array[mask] = new_color

        # Apply median filter to denoise the image
        for i in range(3):  # Apply the filter to the RGB channels, not the alpha channel
            img_array[:, :, i] = median_filter(img_array[:, :, i], size=3)

        # Convert the array back to an Image object
        new_img = Image.fromarray(img_array)

        # Generate the output file name
        output_filename = f"{input_filename.rsplit('.', 1)[0]}_converted.gif"

        # Save the converted and denoised image
        new_img.save(output_filename)

        return output_filename
    except Exception as e:
        return f"An error occurred: {e}"


def plot_success_rates(plot_stat_dict, plot_metrics, th_min=None, th_max=None, file_name=None):
    # Number of models
    num_models = len(plot_stat_dict)

    # Create figure with subplots
    fig, axes = plt.subplots(num_models, len(plot_metrics), figsize=(6 * len(plot_metrics), 5 * num_models), squeeze=False)

    # Adjust layout
    fig.tight_layout(pad=4.0)

    for i, (model_name, methods_data) in enumerate(plot_stat_dict.items()):
        for j, key in enumerate(plot_metrics):
            ax = axes[i, j]

            all_values = np.array([value for method_data in methods_data.values() for value in method_data[key]])
            global_min, global_max = all_values.min(), all_values.max()
            if th_min is not None:
                global_min = max(th_min, global_min * 0.9)
            if th_max is not None:
                global_max = min(th_max, global_max * 1.1)
            # Generate thresholds from global min to max
            thresholds = np.linspace(global_min, global_max, 100)

            for method, data in methods_data.items():
                # Calculate success rates for a range of thresholds
                if key == "cost_seq":
                    max_step = max(len(cost_seq) for cost_seq in data[key])
                    all_indices_list = [[np.where(np.array(cost_seq) < t)[0] for cost_seq in data[key]] for t in thresholds]
                    fst_indices_list = [[indices[0] if indices.size > 0 else max_step for indices in t_list]for t_list in all_indices_list]
                    success_steps = [np.mean(np.array(fst_indices)) for fst_indices in fst_indices_list]
                    ax.plot(thresholds, success_steps, label=method)
                    pass
                else:
                    success_rates = [np.mean(np.array(data[key]) < t) for t in thresholds]
                    ax.plot(thresholds, success_rates, label=method)

            # Setting the titles and labels
            ax.set_title(f'Model: {model_name} - {key.replace("_", " ").title()}')
            ax.set_xlabel("Threshold")
            if key == "cost_seq":
                ax.set_ylabel("Success Step")
            else:
                ax.set_ylabel("Success Rate")
            ax.legend()
    if file_name is not None:
        plt.savefig(file_name)
        print(f"Plot saved to {file_name}")
        plt.close(fig)
    else:
        plt.show()


def plot_prediction_error_in_planning(all_res, num_test, file_name):
    diff_statistics = {}
    rmse_statistics = {}
    for model_name, cases in all_res.items():
        cost_diff_dict = {}
        gt_state_dict = {}
        pred_state_dict = {}
        i = 0
        for case, res in cases.items():
            if i >= num_test:
                break
            i += 1
            if diff_statistics == {}:
                method_list = res["methods"]
                for method in method_list:
                    diff_statistics[method] = {}
                    rmse_statistics[method] = {}
            for method in method_list:
                if method not in cost_diff_dict:
                    cost_diff_dict[method] = []
                    gt_state_dict[method] = []
                    pred_state_dict[method] = []
                cost_diff_dict[method].append(res[method]["result_summary"]["cost_diff_seq"])
                gt_state_dict[method].append(res[method]["result_summary"]["gt_states"])
                pred_state_dict[method].append(res[method]["result_summary"]["pred_state_seq"])
        for method in cost_diff_dict:
            cost_diff = np.array(cost_diff_dict[method])
            cost_diff_sq = cost_diff**2
            diff_statistics[method][model_name] = {
                "mean": np.concatenate(([0], np.median(cost_diff_sq, axis=0))),
                "percentile_25": np.concatenate(([0], np.percentile(cost_diff_sq, 25, axis=0))),
                "percentile_75": np.concatenate(([0], np.percentile(cost_diff_sq, 75, axis=0))),
            }
            pred_states = np.array(pred_state_dict[method])
            gt_states = np.array(gt_state_dict[method])[:, 1:, : pred_states.shape[2]]
            rmse_data = np.sqrt(np.mean(((gt_states - pred_states) / 100) ** 2, axis=2))
            rmse_statistics[method][model_name] = {
                "mean": np.concatenate(([0], np.median(rmse_data, axis=0))),
                "percentile_25": np.concatenate(([0], np.percentile(rmse_data, 25, axis=0))),
                "percentile_75": np.concatenate(([0], np.percentile(rmse_data, 75, axis=0))),
            }
    # Create a subplot for each method in 2 rows
    n_methods = len(diff_statistics)
    fig, axs = plt.subplots(2, n_methods, figsize=(n_methods * 5, 10))  # Adjust the size as needed
    for i, method in enumerate(diff_statistics):
        ax_diff = axs[0, i] if n_methods > 1 else axs[0]  # First row for diff_statistics
        ax_rmse = axs[1, i] if n_methods > 1 else axs[1]  # Second row for rmse_statistics

        plot_open_pred_stat(diff_statistics[method], len(cost_diff[0]) + 1, None, ax_diff, method + "- Cost Diff")
        plot_open_pred_stat(rmse_statistics[method], len(cost_diff[0]) + 1, None, ax_rmse, method + "- RMSE")

    plt.tight_layout()
    fig.suptitle("Prediction Error among Different Methods")
    plt.savefig(file_name)
    plt.close(fig)
    print(f"Plot saved to {file_name}")


def plot_planning_latent(res, open_loop, file_name, params, ae_model):
    planning_methods = res["methods"]
    ae_model = ae_model.eval().to("cpu")
    max_steps = max(len(res[method]["result_summary"]["gt_states"]) for method in planning_methods)
    state_dim = ae_model.config["latent"]["latent_dim"]
    for method in planning_methods:
        gt_states = res[method]["result_summary"]["gt_states"]
        pred_states = res[method]["result_summary"]["pred_state_seq"]
        gt_states = torch.tensor(gt_states, dtype=torch.float32)[:, :state_dim]
        pred_states = torch.tensor(pred_states, dtype=torch.float32)[:, :state_dim]
        pred_states = torch.cat([gt_states[0:1], pred_states], dim=0)
        # [max_steps, state_dim]
        gt_images = ae_model.decoder(gt_states).detach().numpy().transpose(0, 2, 3, 1) * 255
        pred_images = ae_model.decoder(pred_states).detach().numpy().transpose(0, 2, 3, 1) * 255
        frames = [np.concatenate([gt_images[i], pred_images[i]], axis=1) for i in range(max_steps)]
        save_gif(frames, file_name + f"_{method}.gif")
    ae_model = ae_model.to("cuda")
    return


def plot_long_for_hier(res_list, exp_setting, file_name, params, ae_model=None):
    task_name = params["task_name"]
    if "latent" in task_name:
        raise NotImplementedError("Latent space tasks are not supported for hierarchical planning")
        assert ae_model is not None, "AE model must be provided for latent space tasks"
        return plot_planning_latent(res, open_loop, file_name, params, ae_model)
    object_type = convert_task_to_obj(task_name)
    assert object_type in ["box", "T", "L"], "object_type must be one of 'box', 'T', or 'L'"
    window_size = params["window_size"]
    workspace_size = (window_size, window_size)
    target_pose = exp_setting["target_pose"]
    planning_methods = exp_setting["methods"]
    max_steps = exp_setting["horizon"] + 1
    fig, axes = plt.subplots(len(planning_methods), figsize=(5, 5 * len(planning_methods)))
    if len(planning_methods) == 1:
        axes = [axes]
    res = {}
    for i, method in enumerate(planning_methods):
        res[method] = res_list[i]
    anim = FuncAnimation(
        fig,
        update_plot,
        frames=max_steps,
        fargs=(
            res,
            axes,
            planning_methods,
            True,
            workspace_size,
            target_pose,
            object_type,
            params,
            True
        ),
        repeat=False,
    )
    anim.save(f"{file_name}.gif", writer=PillowWriter(fps=max(max_steps//5, 1)))
    # anim.save(f"{file_name}.mp4", FFMpegWriter(fps=max(max_steps // 5, 1)))
    print(f"Plot saved to {file_name}.gif")
    plt.close(fig)

def plot_planning(res, open_loop, file_name, params, ae_model=None):
    task_name = params["task_name"]
    if "latent" in task_name:
        assert ae_model is not None, "AE model must be provided for latent space tasks"
        return plot_planning_latent(res, open_loop, file_name, params, ae_model)
    object_type = convert_task_to_obj(task_name)
    assert object_type in ["box", "T", "L"], "object_type must be one of 'box', 'T', or 'L'"
    window_size = params["window_size"]
    workspace_size = (window_size, window_size)
    init_pose = res["init_pose"]
    target_pose = res["target_pose"]
    init_pusher_pos = res["init_pusher_pos"]
    planning_methods = res["methods"]

    # pred_state_seq, gt_states
    max_steps = max(len(res[method]["result_summary"]["gt_states"]) for method in planning_methods)
    fig, axes = plt.subplots(len(planning_methods), 2, figsize=(5*2, 5 * len(planning_methods)))
    axes = axes.reshape(-1, 2)
    anim = FuncAnimation(
        fig,
        update_plot,
        frames=max_steps,
        fargs=(
            res,
            axes,
            planning_methods,
            open_loop,
            workspace_size,
            target_pose,
            object_type,
            params,
        ),
        repeat=False,
    )
    anim.save(f"{file_name}.gif", writer=PillowWriter(fps=max(max_steps//6, 1)))
    # anim.save(f"{file_name}.mp4", FFMpegWriter(fps=max(max_steps // 15, 1)))
    print(f"Plot saved to {file_name}.gif")
    plt.close(fig)


def update_plot(step_id, res, axes, planning_methods, open_loop, workspace_size, target_pose, object_type, params, gt_only=False):
    state_dim = params["state_dim"]
    min_alpha = 0.2
    for i, method in enumerate(planning_methods):
        # import pdb; pdb.set_trace()
        if gt_only:
            ax_gt = axes[i]
        else:
            ax_gt = axes[i, 0]
            ax_pred = axes[i, 1]
        ax_gt.clear()
        ax_gt.set_xlim(0, workspace_size[0])
        ax_gt.set_ylim(0, workspace_size[1])
        ax_gt.set_aspect("equal", adjustable="box")
        ax_gt.set_title(f"{method} - {'Open' if open_loop else 'Closed'} Loop Planning - Step {step_id} - GT")
        if not gt_only:
            ax_pred.clear()
            ax_pred.set_xlim(0, workspace_size[0])
            ax_pred.set_ylim(0, workspace_size[1])
            ax_pred.set_aspect("equal", adjustable="box")
            ax_pred.set_title(f"{method} - {'Open' if open_loop else 'Closed'} Loop Planning - Step {step_id} - Pred")

        obs_pos_list = res[method]["exp_setting"].get("obs_pos_list", None)
        if obs_pos_list is not None:
            obs_type = res[method]["exp_setting"]["obs_type"]
            assert obs_type in ["circle", "square"], "obs_type must be one of 'circle' or 'square'"
            obs_size_list = res[method]["exp_setting"]["obs_size_list"]
            scale = res[method]["exp_setting"]["scale"]
            obs_pos_list = np.array(obs_pos_list) * scale
            obs_size_list = np.array(obs_size_list) * scale
            obs_enlarge = res[method]["exp_setting"]["obs_enlarge"]
            draw_obs(ax_gt, obs_pos_list, obs_size_list, obs_type, obs_enlarge=0)
            if not gt_only:
                draw_obs(ax_pred, obs_pos_list, obs_size_list, obs_type, obs_enlarge=0)
        # Draw target object with full opacity
        pusher_size = params["pusher_size"]
        if object_type == "box":
            target_keypoints = box_sim.get_keypoints_from_pose(target_pose[0], params)
            draw_object_and_pusher(ax_gt, "box", target_keypoints, color="r", alpha=1.0)
            if not gt_only:
                draw_object_and_pusher(ax_pred, "box", target_keypoints, color="r", alpha=1.0)
        elif object_type == "T":
            target_keypoints = t_sim.get_keypoints_from_pose(target_pose[0], params)
            draw_object_and_pusher(ax_gt, "T", target_keypoints, color="tomato", alpha=1.0)
            if not gt_only:
                draw_object_and_pusher(ax_pred, "T", target_keypoints, color="tomato", alpha=1.0)
        elif object_type == "L":
            target_keypoints_0 = l_sim.get_keypoints_from_pose(target_pose[0], params)
            draw_object_and_pusher(ax_gt, "L", target_keypoints_0, color="lightblue")
            if not gt_only:
                draw_object_and_pusher(ax_pred, "L", target_keypoints_0, color="lightblue")
            target_keypoints_1 = l_sim.get_keypoints_from_pose(target_pose[1], params)
            draw_object_and_pusher(ax_gt, "L", target_keypoints_1, color="yellow")
            if not gt_only:
                draw_object_and_pusher(ax_pred, "L", target_keypoints_1, color="yellow")
        elif object_type == "HP":
            target_keypoints_0 = hp_sim.get_keypoints_from_pose(target_pose[0], params, "hole")
            draw_object_and_pusher(ax_gt, "H", target_keypoints_0, color="g")
            if not gt_only:
                draw_object_and_pusher(ax_pred, "H", target_keypoints_0, color="g")
            target_keypoints_1 = hp_sim.get_keypoints_from_pose(target_pose[1], params, "peg")
            draw_object_and_pusher(ax_gt, "P", target_keypoints_1, color="r")
            if not gt_only:
                draw_object_and_pusher(ax_pred, "P", target_keypoints_1, color="r")

        # Draw current T object and pusher from gt_states if available
        gt_states = res[method]["result_summary"]["gt_states"]
        
        if open_loop:
            end_step = len(gt_states)
        else:
            end_step = step_id + 1
        
        for i in range(step_id, end_step):
            current_state = gt_states[i]
            current_pusher_pos = current_state[state_dim: state_dim + 2]
            alpha = max(1.0 - 1 * ((i - step_id) / (end_step - step_id)), min_alpha)
            draw_one_step(ax_gt, object_type, current_state[:state_dim], current_pusher_pos, pusher_size, color="b", alpha=alpha)
            if i == step_id:
                init_pusher_pos = current_pusher_pos
        if gt_only:
            continue
        all_res_states = res[method]["all_res"]
        num_planning = len(all_res_states)
        pred_states = res[method]["result_summary"]["pred_state_seq"]
        if step_id <= len(pred_states):
            for i in range(num_planning):
                plan_start_step = all_res_states[i]["start_step"]
                if step_id < plan_start_step:
                    plan_start_step = all_res_states[i - 1]["start_step"]
                    i -= 1
                    break
            planned_states = all_res_states[i]["state_seq"]
            act_seq = all_res_states[i]["act_seq"]
            # import pdb; pdb.set_trace()

            planned_states = np.concatenate([[gt_states[step_id][:state_dim]], planned_states])
            act_seq = np.concatenate([act_seq, np.zeros((1, 2))])
            planned_pusher_pos = np.array(init_pusher_pos, dtype=np.float32)
            num_visible_steps = len(planned_states) - (step_id - plan_start_step)
            # print(f"Step {step}: {num_visible_steps} steps visible")
            states_to_plot = planned_states[-num_visible_steps:]
            actions_to_plot = act_seq[-num_visible_steps:]
            # import pdb; pdb.set_trace()
            for i, (state, action) in enumerate(zip(states_to_plot, actions_to_plot)):
                # alpha = 0.7 - 0.35 * ((i + 1) / num_visible_steps)
                alpha = max(1.0 - 1 * ((i) / max(num_visible_steps,1)), min_alpha)
                # Update pusher position based on action
                # planned_pusher_pos += np.array(action)
                draw_one_step(ax_pred, object_type, state[:state_dim], planned_pusher_pos, pusher_size, color="c", alpha=alpha)
                planned_pusher_pos += np.array(action)

def draw_obs(ax, obs_pos_list, obs_size_list, obs_type, obs_enlarge=0):    
    for obs_pos, obs_size in zip(obs_pos_list, obs_size_list):
        if obs_enlarge > 0:
            enlarged_size = (1 + obs_enlarge) * obs_size
            if obs_type == "circle":
                ax.add_patch(plt.Circle(obs_pos, enlarged_size, color="grey", alpha=1.0))
            elif obs_type == "square":
                # draw a square centered at obs_pos with size 2*obs_size
                ax.add_patch(plt.Rectangle((obs_pos[0] - enlarged_size, obs_pos[1] - enlarged_size), 2*enlarged_size, 2*enlarged_size, color="red", alpha=1.0))
        if obs_type == "circle":
            ax.add_patch(plt.Circle(obs_pos, obs_size, color="grey", alpha=1.0))
        elif obs_type == "square":
            # draw a square centered at obs_pos with size 2*obs_size
            ax.add_patch(plt.Rectangle((obs_pos[0] - obs_size, obs_pos[1] - obs_size), 2*obs_size, 2*obs_size, color="grey", alpha=1.0))
    return

def draw_one_step(ax, object_type, current_state, current_pusher_pos, pusher_size=5, color="b", alpha=1.0):
    state_dim = len(current_state)
    if object_type == "box" or object_type == "T":
        num_kp = state_dim // 2
        current_keypoints = np.array(current_state).reshape((num_kp, 2))
        draw_object_and_pusher(
            ax,
            object_type,
            current_keypoints,
            current_pusher_pos, 
            pusher_size,
            color="orange",
            alpha=alpha,
        )
    elif object_type == "L":
        num_kp = state_dim // 2 // 2
        current_keypoints = np.array(current_state[:num_kp * 2]).reshape((num_kp, 2))
        draw_object_and_pusher(
            ax,
            object_type,
            current_keypoints,
            current_pusher_pos,
            pusher_size,
            color="b",
            alpha=alpha,
        )
        current_keypoints = np.array(current_state[num_kp * 2: state_dim]).reshape((num_kp, 2))
        draw_object_and_pusher(ax, object_type, current_keypoints, color="orange", alpha=alpha)
    elif object_type == "HP":
        if state_dim == 18:
            num_kp_hole = 5
            num_kp_peg = 4
        elif state_dim == 22:
            num_kp_hole = 6
            num_kp_peg = 5
        current_keypoints = np.array(current_state[:num_kp_hole*2]).reshape((num_kp_hole, 2))
        draw_object_and_pusher(
            ax,
            "H",
            current_keypoints,
            current_pusher_pos,
            pusher_size,
            color="b",
            alpha=alpha,
        )
        current_keypoints = np.array(current_state[num_kp_hole*2: state_dim]).reshape((num_kp_peg, 2))
        draw_object_and_pusher(ax, "P", current_keypoints, color="orange", alpha=alpha)

def draw_object_and_pusher(ax, object_type, keypoints, pusher=None, pusher_size=5, color="b", alpha=1.0, pusher_color='k'):
    if object_type == "box":
        for i in range(4):
            next_i = (i + 1) % 4
            ax.plot(
                [keypoints[i][0], keypoints[next_i][0]], [keypoints[i][1], keypoints[next_i][1]], c=color, alpha=alpha
            )
    elif object_type == "T":
        keypoints = np.array(keypoints)
        # keypoints = 500 - keypoints
        if pusher is not None:
            pusher = np.array(pusher)
            # pusher = 500 - pusher
        # solid T
        left_point, middle_point, right_point, end_point = keypoints
        bar_vector = left_point - right_point
        stem_vector = middle_point - end_point
        bar_vector /= np.linalg.norm(bar_vector)
        stem_vector /= np.linalg.norm(stem_vector)
        
        bar_thickness = 30 / 2
        stem_thickness = 30 / 2
        rect1 = plt.Polygon([left_point+stem_vector*bar_thickness, 
                            left_point-stem_vector*bar_thickness,  
                            right_point-stem_vector*bar_thickness,
                            right_point+stem_vector*bar_thickness], 
                            closed=True, fill=True, edgecolor=color, 
                            facecolor=color, alpha=alpha)
        ax.add_patch(rect1)

        rect2 = plt.Polygon([middle_point+bar_vector*stem_thickness, 
                            middle_point-bar_vector*stem_thickness, 
                            end_point-bar_vector*stem_thickness, 
                            end_point+bar_vector*stem_thickness], 
                            closed=True, fill=True, edgecolor=color, 
                            facecolor=color, alpha=alpha)
        ax.add_patch(rect2)
        # stick T
        # ax.plot([keypoints[0][0], keypoints[1][0]], [keypoints[0][1], keypoints[1][1]], c=color, alpha=alpha)
        # ax.plot([keypoints[1][0], keypoints[2][0]], [keypoints[1][1], keypoints[2][1]], c=color, alpha=alpha)
        # ax.plot([keypoints[1][0], keypoints[3][0]], [keypoints[1][1], keypoints[3][1]], c=color, alpha=alpha)
    elif object_type == "L":
        keypoints = np.array(keypoints)
        keypoints = 500 - keypoints
        if pusher is not None:
            pusher = np.array(pusher)
            pusher = 500 - pusher
        left_L = keypoints[:3]
        right_L = keypoints[3:]
        unit_size = 45
        for i, L_points in enumerate([left_L, right_L]):
            leg_point, middle_point, foot_point = L_points
            leg_vector = leg_point - middle_point
            foot_vector = foot_point - middle_point
            leg_vector /= np.linalg.norm(leg_vector)
            foot_vector /= np.linalg.norm(foot_vector)
            thickness = unit_size / 2
            # Draw the horizontal part of the T (using points 1 and 2)
            leg = plt.Polygon([leg_point+foot_vector*thickness, 
                        leg_point-foot_vector*thickness,  
                        middle_point-leg_vector*thickness-foot_vector*thickness,
                        middle_point-leg_vector*thickness+foot_vector*thickness],
                        closed=True, fill=True, edgecolor=color[i],
                        facecolor=color[i], alpha=alpha) 
            ax.add_patch(leg)
            # Draw the vertical part of the T (using points 3 and 4)
            foot = plt.Polygon([foot_point+leg_vector*thickness, 
                        foot_point-leg_vector*thickness, 
                        middle_point-leg_vector*thickness, 
                        middle_point+leg_vector*thickness],
                        closed=True, fill=True, edgecolor=color[i],
                        facecolor=color[i], alpha=alpha)
            ax.add_patch(foot)
    elif object_type == "H":
        # Draw hole, 0->1->2->3->4
        for i in range(4):
            next_i = i + 1
            ax.plot(
                [keypoints[i][0], keypoints[next_i][0]], [keypoints[i][1], keypoints[next_i][1]], c=color, alpha=alpha
            )
    elif object_type == "P":
        # Draw peg, 0->1->2, 3->1
        for i in range(2):
            next_i = i + 1
            ax.plot(
                [keypoints[i][0], keypoints[next_i][0]], [keypoints[i][1], keypoints[next_i][1]], c=color, alpha=alpha
            )
        ax.plot([keypoints[3][0], keypoints[1][0]], [keypoints[3][1], keypoints[1][1]], c=color, alpha=alpha)
    # Draw com if there is
    if object_type == "H" and len(keypoints) == 6:
        ax.scatter(keypoints[5][0], keypoints[5][1], c="black", s=5, alpha=alpha)
    elif object_type != "H" and len(keypoints) == 5:
        ax.scatter(keypoints[4][0], keypoints[4][1], c="black", s=5, alpha=alpha)
    elif object_type == "L" and len(keypoints) == 4:
        ax.scatter(keypoints[3][0], keypoints[3][1], c="black", s=5, alpha=alpha)
    # Draw pusher if position is provided
    if pusher is not None:
        ax.scatter(pusher[0], pusher[1], c=pusher_color, s=pusher_size**2, alpha=alpha)

# def draw_object_and_pusher(ax, object_type, keypoints, pusher=None, pusher_size=5, color="b", alpha=1.0, pusher_color='k'):
#     if object_type == "box":
#         for i in range(4):
#             next_i = (i + 1) % 4
#             ax.plot(
#                 [keypoints[i][0], keypoints[next_i][0]], [keypoints[i][1], keypoints[next_i][1]], c=color, alpha=alpha
#             )
#     elif object_type == "T":
#         ax.plot([keypoints[0][0], keypoints[1][0]], [keypoints[0][1], keypoints[1][1]], c=color, alpha=alpha)
#         ax.plot([keypoints[1][0], keypoints[2][0]], [keypoints[1][1], keypoints[2][1]], c=color, alpha=alpha)
#         ax.plot([keypoints[1][0], keypoints[3][0]], [keypoints[1][1], keypoints[3][1]], c=color, alpha=alpha)
#     elif object_type == "L":
#         for i in range(len(keypoints)-1):
#             next_i = i + 1
#             ax.plot(
#                 [keypoints[i][0], keypoints[next_i][0]], [keypoints[i][1], keypoints[next_i][1]], c=color, alpha=alpha)
            
#         # if len(keypoints) == 4:
#         #     ax.plot([keypoints[0][0], keypoints[1][0]], [keypoints[0][1], keypoints[1][1]], c=color, alpha=alpha)
#         #     ax.plot([keypoints[1][0], keypoints[2][0]], [keypoints[1][1], keypoints[2][1]], c=color, alpha=alpha)
#         #     ax.plot([keypoints[2][0], keypoints[3][0]], [keypoints[2][1], keypoints[3][1]], c=color, alpha=alpha)
#         # # elif len(keypoints) == 3:
#         # #     ax.plot([keypoints[0][0], keypoints[1][0]], [keypoints[0][1], keypoints[1][1]], c=color, alpha=alpha)
#         # #     ax.plot([keypoints[1][0], keypoints[2][0]], [keypoints[1][1], keypoints[2][1]], c=color, alpha=alpha)
#     elif object_type == "H":
#         # Draw hole, 0->1->2->3->4
#         for i in range(4):
#             next_i = i + 1
#             ax.plot(
#                 [keypoints[i][0], keypoints[next_i][0]], [keypoints[i][1], keypoints[next_i][1]], c=color, alpha=alpha
#             )
#     elif object_type == "P":
#         # Draw peg, 0->1->2, 3->1
#         for i in range(2):
#             next_i = i + 1
#             ax.plot(
#                 [keypoints[i][0], keypoints[next_i][0]], [keypoints[i][1], keypoints[next_i][1]], c=color, alpha=alpha
#             )
#         ax.plot([keypoints[3][0], keypoints[1][0]], [keypoints[3][1], keypoints[1][1]], c=color, alpha=alpha)
#     # Draw com if there is
#     if object_type == "H" and len(keypoints) == 6:
#         ax.scatter(keypoints[5][0], keypoints[5][1], c="black", s=5, alpha=alpha)
#     elif object_type != "H" and len(keypoints) == 5:
#         ax.scatter(keypoints[4][0], keypoints[4][1], c="black", s=5, alpha=alpha)
#     elif object_type == "L" and len(keypoints) == 4:
#         ax.scatter(keypoints[3][0], keypoints[3][1], c="black", s=5, alpha=alpha)
#     # Draw pusher if position is provided
#     if pusher is not None:
#         ax.scatter(pusher[0], pusher[1], c=pusher_color, s=pusher_size**2, alpha=alpha)


def plot_convergence_model(planning_results, inter_results, method_types, max_iterations, file_name):
    """
    Creates a plot with subplots for each model, comparing CROWN with MPPI and MIP.

    Parameters:
    - planning_results: Final results for MPPI and MIP.
    - inter_results: Intermediate results from CROWN.
    - max_iterations: Maximum number of iterations to plot.
    """
    inter_results = padding_result(inter_results, max_iterations)
    model_list = list(planning_results.keys())
    num_models = len(model_list)
    grid_size = math.ceil(math.sqrt(num_models))
    num_rows = grid_size if num_models > grid_size * (grid_size - 1) else grid_size - 1
    num_columns = grid_size

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_columns * 4, num_rows * 3.5))
    if num_columns == 1 and num_rows == 1:
        axs = np.array([axs])  # Ensure axs is always a list for consistency
    axs = axs.flatten()

    for i, model_name in enumerate(model_list):
        draw_model_subplot(axs[i], model_name, inter_results, planning_results, max_iterations, method_types)

    # # Hide any unused subplots
    # for ax in axs[num_models:]:
    #     ax.set_visible(False)

    fig.suptitle("Planned Objective over Iterations for Models")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close(fig)
    print(f"Plot saved to {file_name}")
    # plt.show()


def draw_model_subplot(ax, model_name, inter_results, planning_results, max_iterations, method_types):
    """
    Draws a subplot for a specific model comparing CROWN with MPPI and MIP.

    Parameters:
    - ax: Axes to plot on.
    - model_name: Name of the model.
    - inter_results: Intermediate results from CROWN.
    - planning_results: Final results for MPPI and MIP.
    - max_iterations: Maximum number of iterations to plot.
    """
    colors = [
        "skyblue",
        "sandybrown",
        "lightgreen",
        "pink",
        "lightgrey",
        "lightcoral",
        "lightskyblue",
        "lightsteelblue",
    ]
    # Plot CROWN convergence curve with percentiles
    if model_name in inter_results:
        objective_array = inter_results[model_name]
        accumulated_objective = np.sum(objective_array, axis=1)
        mean_objective = np.median(accumulated_objective, axis=0)
        percentile_25 = np.percentile(accumulated_objective, 25, axis=0)
        percentile_75 = np.percentile(accumulated_objective, 75, axis=0)

        iteration_indices = np.arange(0, max_iterations+1)
        color = "blue"
        ax.plot(iteration_indices, mean_objective, label="CROWN", color=color, linewidth=1.5)
        ax.fill_between(iteration_indices, percentile_25, percentile_75, color=color, edgecolor=color, alpha=0.2)
        ax.plot(iteration_indices, percentile_25, alpha=0.5, color=color, linewidth=0.5)
        ax.plot(iteration_indices, percentile_75, alpha=0.5, color=color, linewidth=0.5)

    # Plot median values for MPPI and MIP
    linestyle = "dashed"
    colors = ["red", "green", "purple", "yellow"]
    i = 0
    if model_name in planning_results:
        for method in method_types:
            if method not in planning_results[model_name]["planned_cost"] or method == "CROWN":
                continue
            color = colors[i%len(colors)]
            i += 1
            method_results = planning_results[model_name]["planned_cost"][method]
            method_median = np.median(method_results)
            method_percentile_25 = np.percentile(method_results, 25)
            method_percentile_75 = np.percentile(method_results, 75)

            ax.hlines(method_median, 0, max_iterations, label=f"{method}", color=color, linestyles=linestyle)
            ax.hlines(method_percentile_25, 0, max_iterations, color=color, linestyles=linestyle, alpha=0.5)
            ax.hlines(method_percentile_75, 0, max_iterations, color=color, linestyles=linestyle, alpha=0.5)

    # Set x-axis ticks
    tick_interval = max(max_iterations // 5, 1)
    ax.set_xticks(np.arange(0, max_iterations + 1, tick_interval))

    ax.set_title(f"Model: {model_name}")
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Planned Objective")
    ax.legend()


def padding_result(inter_results, max_iterations):
    max_iterations = max_iterations+1
    # Handle different number of iterations
    for model_name in inter_results:
        # Padding each innermost list to have the same length
        padded_data = [
            [step[: min(max_iterations, len(step))] + [np.nan] * (max_iterations - len(step)) for step in case]
            for case in inter_results[model_name]
        ]
        inter_results[model_name] = np.array(padded_data[:][:][:max_iterations])
    return inter_results


# plot function for open or closed loop planning
def plot_planning_stat(planning_results, file_name, outlier_threshold=None, lb=None, ub=None):
    model_list = list(planning_results.keys())
    cost_types = list(planning_results[model_list[0]].keys())
    if len(cost_types) == 3 or len(cost_types) == 4:
        nrows, ncols = 2, 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 9))
    elif len(cost_types) == 5 or len(cost_types) == 6:
        nrows, ncols = 2, 3
        fig, axs = plt.subplots(nrows, ncols, figsize=(18, 9))
    else:
        raise ValueError("Invalid number of cost types")
    method_types = list(planning_results[model_list[0]][cost_types[0]].keys())
    colors = [
        "skyblue",
        "sandybrown",
        "lightgreen",
        "pink",
        "lightgrey",
        "lightcoral",
        "lightskyblue",
        "lightsteelblue",
    ]

    for subplot_index, cost_type in enumerate(cost_types):
        ax = axs[subplot_index // ncols][subplot_index % ncols]
        draw_box_plots(ax, planning_results, model_list, cost_type, method_types, colors, outlier_threshold, lb, ub)

    base_name = os.path.basename(file_name)
    title = os.path.splitext(base_name)[0]
    fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close(fig)
    print(f"Plot saved to {file_name}")


def draw_box_plots(
    ax, planning_results, model_list, cost_type, method_types, colors, outlier_threshold=None, lb=None, ub=None
):
    """
    Draws box plots on the given axes for the specified cost type.

    Parameters:
    - ax: The axes on which to draw the box plots.
    - planning_results: Data containing planning results.
    - model_list: List of model names.
    - cost_type: Type of cost to plot.
    - method_types: List of method types.
    - colors: List of colors for the box plots.
    """
    data_for_plotting = []

    for model_name in model_list:
        if cost_type in planning_results[model_name]:
            for method in method_types:
                if lb is not None and ub is not None:
                    filtered_data = filter_outliers(planning_results[model_name][cost_type][method], None, lb, ub)
                else:
                    filtered_data = filter_outliers(planning_results[model_name][cost_type][method], outlier_threshold)
                data_for_plotting.append(filtered_data)

    # Calculating positions dynamically
    width = 0.15  # Width of each boxplot
    positions = []

    for index, model_name in enumerate(model_list, start=1):
        start_pos = index - width * len(method_types) / 2
        method_positions = [start_pos + j * width for j in range(len(method_types))]
        positions.extend(method_positions)

    for i in range(0, len(data_for_plotting), len(method_types)):
        for j in range(len(method_types)):
            ax.boxplot(
                data_for_plotting[i + j],
                positions=[positions[i + j]],
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor=colors[j % len(colors)]),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                showfliers=True,
            )

    ax.set_xticks(np.arange(1, len(model_list) + 1))
    ax.set_xticklabels(model_list)
    ax.set_xlabel("Models")
    ax.set_ylabel("Objective Value")
    ax.set_title(f"{cost_type}")

    legend_elements = [
        Patch(facecolor=colors[i % len(colors)], label=method_types[i]) for i in range(len(method_types))
    ]
    ax.legend(handles=legend_elements, loc="upper right")


def filter_outliers(data, threshold=None, lower_bound=None, upper_bound=None):
    """
    Filters outliers that are beyond the specified threshold in the IQR.

    Parameters:
    - data: A list of data points.
    - threshold: The multiplier for the IQR to determine outliers. Default is 1.5.

    Returns:
    - A filtered list of data points without extreme outliers.
    """
    if threshold is None and (lower_bound is None or upper_bound is None):
        return data
    if threshold is not None:
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (iqr * threshold)
        upper_bound = q3 + (iqr * threshold)
    return [x for x in data if lower_bound <= x <= upper_bound]


# plot function to visualize the open loop prediction
def plot_open_pred_stat(rmse_statistics, len_episode, img_name=None, ax=None, title=None):
    assert img_name is None or ax is None, "Cannot save image if ax is provided"
    # Create a figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Generate evenly spaced time steps, assuming len_episode - 1 is a multiple of 5
    time_steps = np.arange(0, len_episode, 5)

    model_list = list(rmse_statistics.keys())
    # colors = cm.viridis(np.linspace(0, 1, len(model_list)))
    # rainbow colors
    colors = cm.rainbow(np.linspace(0, 1, len(model_list)))

    # Plot RMSE statistics for each model
    for model_name, color in zip(model_list, colors):
        stats = rmse_statistics[model_name]
        # Plot the mean RMSE
        ax.plot(range(len_episode), stats["mean"], label=f"{model_name}", color=color, linewidth=2.5)
        # Shade the area between the 25th and 75th percentiles
        ax.fill_between(
            range(len_episode), stats["percentile_25"], stats["percentile_75"], color=color, edgecolor=color, alpha=0.1
        )
        ax.plot(range(len_episode), stats["percentile_25"], color=color, alpha=0.5, linewidth=0.5)
        ax.plot(range(len_episode), stats["percentile_75"], color=color, alpha=0.5, linewidth=0.5)

    # Set the x-axis ticks to be integers that are multiples of 5
    ax.set_xticks(time_steps)
    ax.set_xticklabels(time_steps)

    # Add legend and labels
    ax.set_xlabel("Timestep")
    ax.set_ylabel("RMSE")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("RMSE over Time for Models with Different Models")
    ax.legend()

    # Save the figure
    if img_name is not None:
        plt.savefig(img_name)
        print(f"Plot saved to {img_name}")
        plt.close(fig)


# plot function to visualize a demo of open loop prediction for box or T task


def plot_open_pred_demo(episodes_dict, file_name, param_dict):
    """
    Visualize a dictionary of episodes of states in one figure with multiple subplots.

    Parameters:
        episodes_dict (dict): Dictionary where key is the model name and value is a numpy array
                              of shape (num_steps, state_dim) representing the episode.
                              state_dim = 10 for box (8 for keypoints, 2 for pusher)
                              state_dim = 10 for T (8 for keypoints, 2 for pusher)
        file_name (str): Name of the output image file.
        param_dict
    """
    object_type = convert_task_to_obj(param_dict["task_name"])
    pusher_size = param_dict["pusher_size"]
    num_episodes = len(episodes_dict)
    grid_size = math.ceil(math.sqrt(num_episodes))
    num_rows = grid_size if num_episodes > grid_size * (grid_size - 1) else grid_size - 1
    num_columns = grid_size

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_columns * 4, num_rows * 3.5))
    if num_columns == 1 and num_rows == 1:
        axs = np.array([axs])  # Ensure axs is always a list for consistency
    axs = axs.flatten()

    for ax, (title, episode_states) in zip(axs, episodes_dict.items()):
        ax.set_title(title)
        length = len(episode_states)
        for i in range(length):
            state = episode_states[i]
            alpha = 1 - 0.7 * ((i) / length)
            if object_type == "box":
                keypoints = state[:8].reshape(4, 2)
                pusher = state[8:10]
                colors = ["r", "g", "b", "y"]
                for i, point in enumerate(keypoints):
                    ax.scatter(point[0], point[1], c=colors[i])
                draw_object_and_pusher(ax, "box", keypoints, pusher, pusher_size, color="m", alpha=alpha)
                for i in range(4):
                    next_i = (i + 1) % 4
                    ax.plot([keypoints[i][0], keypoints[next_i][0]], [keypoints[i][1], keypoints[next_i][1]], c="m", alpha=alpha)
            elif object_type == "T":
                keypoints = state[:8].reshape(4, 2)
                pusher = state[8:10]
                draw_object_and_pusher(ax, "T", keypoints, pusher, pusher_size, color="m", alpha=alpha)
            elif object_type == "L":
                pusher = state[16:18]
                colors = ["b", "y"]
                for i in range(2):
                    keypoints = state[i * 8 : (i + 1) * 8].reshape(4, 2)
                    draw_object_and_pusher(ax, "L", keypoints, pusher, pusher_size, color=colors[i], alpha=alpha)
            elif object_type == "HP":
                pusher = state[18:20]
                draw_object_and_pusher(ax, "H", state[:10].reshape(5, 2), pusher, pusher_size, color="b", alpha=alpha)
                draw_object_and_pusher(ax, "P", state[10:18].reshape(4, 2), None, pusher_size, color="y", alpha=alpha)

            # ax.scatter(pusher[0], pusher[1], c="k")

        ax.axis("equal")
        ax.set_xlim(0, param_dict["window_size"])
        ax.set_ylim(0, param_dict["window_size"])

    # Hide any unused subplots
    for ax in axs[num_episodes:]:
        ax.set_visible(False)

    if file_name is not None:
        plt.savefig(file_name)
        print(f"Plot saved to {file_name}")
        plt.close(fig)
    else:
        plt.show()


# 3 plot functions to visualize the reward in mppi
def plot_mppi_single(res):
    eval_outputs = res["eval_outputs"]
    rew = [out["rewards"].mean().item() for out in eval_outputs]
    plt.plot(rew)
    plt.show()


def plot_mppi_reward(all_res, image_name, save=False):
    """
    Visualize the mean reward across multiple timesteps and update iterations.

    Parameters:
    - all_res: List of 'res' dictionaries obtained from each timestep.
    """
    # Initialize an empty list to hold all reward curves
    all_rew = []

    for res in all_res:
        eval_outputs = res["eval_outputs"]
        # Compute the mean reward for each update iteration in the current timestep
        rew = [out["rewards"].mean().item() for out in eval_outputs]
        all_rew.append(rew)

    # Plotting
    n_timesteps = len(all_rew)
    colors = cm.viridis(np.linspace(0, 1, n_timesteps))
    for i, (rew, color) in enumerate(zip(all_rew, colors)):
        plt.plot(rew, color=color)

    # Adding colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=1, vmax=n_timesteps))
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(0, n_timesteps, 11), label="Timestep")

    plt.xlabel("Update Iteration")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward across All Timesteps")
    if save:
        plt.savefig(image_name, bbox_inches="tight")
    plt.show()


def plot_mppi_best_reward(all_res, image_name, save=False):
    """
    Visualize the best evaluation output (best reward) for each timestep.

    Parameters:
    - all_res: List of 'res' dictionaries obtained from each timestep.
    """
    # Initialize an empty list to hold the best evaluation output for each timestep
    best_eval_outputs = []

    for res in all_res:
        best_eval_output = res["best_eval_output"]
        best_eval_outputs.append(best_eval_output["rewards"].item())
        # best_eval_output = res.get('best_eval_output', None)
        # if best_eval_output is not None:
        #     best_eval_outputs.append(best_eval_output[0].item())  # Assuming best_eval_output is a 1-element tensor

    # Plotting
    plt.plot(best_eval_outputs, marker="o", linestyle="-")

    plt.xlabel("Timestep")
    plt.ylabel("Best Evaluation Output (Best Reward)")
    plt.title("Best Evaluation Output across All Timesteps")
    if save:
        plt.savefig(image_name, bbox_inches="tight")
    plt.show()
