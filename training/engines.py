import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
# import gc
import wandb
from training.dataset import DynamicsDataset
from others.meter import AverageMeter

from util.utils import rigid_transform_2D
# if particles' distance is smaller than interval, then add this penalty
# target: [B, n_roll, state_dim]
def soft_interval_loss(target, interval):
    B, n_roll, state_dim = target.shape
    target = target.reshape(B, n_roll, state_dim//2, 2)
    target_expand1 = target.unsqueeze(2)
    target_expand2 = target.unsqueeze(3)
    diff = target_expand1 - target_expand2
    l2_distance = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-8) # B, n_roll, state_dim//2, state_dim//2
    mask = l2_distance < interval   
    torch.diagonal(mask, dim1=-2, dim2=-1).fill_(False)
    mask = mask.detach()
    penalty = torch.mean((interval - l2_distance[mask]) ** 2)
    return penalty

# box_size: [0.06, 0.015, 0.03]
def box_shape_loss(pred, box_size = [6, 3]):
    B, n_rollout, state_dim = pred.shape
    pred = pred.reshape(B, n_rollout, state_dim//2, 2)
    pred_box_center = torch.mean(pred, dim=-2, keepdim=True) # B, n_rollout, 1, 2
    pred_box_diag_half_len = pred_box_center - pred # B, n_rollout, 4, 2
    pred_box_diag_half_len = torch.norm(pred_box_diag_half_len, dim=-1) # B, n_rollout, 4
    box_diag_half_len = torch.tensor(np.linalg.norm(box_size)).to(pred.device).to(pred.dtype) # 1
    box_diag_half_len = box_diag_half_len.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1,1,1
    box_diag_half_len = box_diag_half_len.repeat(B, n_rollout, 4) # B, n_rollout, 4
    # import pdb; pdb.set_trace()
    loss_shape = F.mse_loss(pred_box_diag_half_len, box_diag_half_len)*0.1
    
    return loss_shape

# box_size: [0.06, 0.015, 0.03]
# keypoints order:[x1,z1], [x1- 0.06*2, z1], [x1 - 0.06*2, z1 - 0.03*2], [x1, z1 - 0.03*2]
# def box_shape_loss(pred, box_size = [0.06, 0.03]):
#     B, n_rollout, state_dim = pred.shape
#     original_box = np.array([[box_size[0], box_size[1]], 
#                             [-box_size[0], box_size[1]], 
#                             [-box_size[0], -box_size[1]], 
#                             [box_size[0], -box_size[1]]])
#     pred = pred.reshape(B, n_rollout, state_dim//2, 2)
#     pred = pred.reshape(B*n_rollout, state_dim//2, 2)
#     original_box = torch.tensor(original_box).to(pred.device).to(pred.dtype)
#     original_box = original_box.unsqueeze(0).repeat(B*n_rollout, 1, 1) # B*n_rollout, 4, 2
#     R, t = rigid_transform_2D(original_box, pred.detach())
#     # import pdb; pdb.set_trace()
#     rotated_box = torch.bmm(original_box, R.transpose(1,2)) + t.transpose(1,2)
#     loss_shape = F.mse_loss(rotated_box, pred)*0.1
    
#     return loss_shape

def validate(data_loader, model, exp, config, epoch, best_valid_loss):
    model.eval()
    MSELoss = nn.MSELoss()
    task_name = config["task_name"]
    train_config = config["train"]
    epochs = train_config["n_epoch_initial"]
    computing_bounds = train_config.get("compute_bounds", False) and epoch == epochs - 1
    is_gnn = config["model"] == "PyG_GNN"
    num_layers = len(train_config["architecture"])
    if "latent" in task_name:
        latent_config = config["latent"]
        use_norm_reg = latent_config["use_norm_reg"]
        norm_type = float(latent_config["norm_type"])
        lam_norm_reg = latent_config["lam_norm_reg"]
    use_ibp_loss = config["train"]["use_ibp_loss"]
    if use_ibp_loss:
        meter_loss_ibp = AverageMeter()
    # set up recording metrics
    meter_loss = AverageMeter()
    meter_loss_rmse = AverageMeter()
    meter_loss_reg = AverageMeter()
    
    inputs = []
    activations = []
    
    for i, data in enumerate(data_loader):
        loss_container = dict()  # store the losses for this step
        with torch.no_grad():
            # n_his, n_roll = train_config["n_history"], train_config["n_rollout"]
            # n_roll = train_config["n_rollout_valid"] if "n_rollout_valid" in train_config else n_roll * 2
            # n_samples = n_his + n_roll # trajectory length
            n_his = data_loader.dataset.n_his
            n_roll = data_loader.dataset.n_roll
            n_samples = data_loader.dataset.n_sample
            # [B, n_samples, obs_dim]
            observations = data["observations"]
            # [B, n_samples, action_dim]
            actions = data["actions"]
            pusher_pos = data["pusher_pos"]
            B = actions.shape[0]
            # [B, n_roll, 1]
            weights = data["weights"].unsqueeze(-1)
            
            if torch.cuda.is_available():
                observations = observations.cuda()
                actions = actions.cuda()
                weights = weights.cuda()
                pusher_pos = pusher_pos.cuda()

            # states, actions = data
            assert actions.shape[1] == n_samples
            loss_mse = 0.0
            # we don't have any visual observations, so states are observations
            states = observations.float() # [B, n_samples, obs_dim]
            # [B, n_his, state_dim]
            state_init = states[:, :n_his, : model.state_dim]
            
            # We want to rollout n_roll steps
            action_seq = actions[:, :-1].float() # [B, n_his+n_roll-1, action_dim]
            input_dict = {"state_init": state_init, "action_seq": action_seq, "pusher_pos": pusher_pos,}
            # [B, n_roll, state_dim]
            state_rollout_gt = states[:, n_his:]
            # try using models_dy.rollout_model instead of doing this manually
            verbose = computing_bounds or use_ibp_loss
            rollout_data = model.rollout_model(
                input_dict=input_dict, verbose=verbose
            )

            # [B, n_roll, state_dim]
            state_rollout_pred = rollout_data["state_pred"]

            if (not is_gnn) and verbose:
                if inputs == []:
                    inputs = rollout_data["input"]
                    activations = rollout_data["activation"]
                else:
                    for rollout_idx in range(n_roll):
                        for layer_idx in range(num_layers):
                            inputs[rollout_idx][layer_idx][0] = np.minimum(
                                inputs[rollout_idx][layer_idx][0],
                                rollout_data["input"][rollout_idx][layer_idx][0],
                            )
                            inputs[rollout_idx][layer_idx][1] = np.maximum(
                                inputs[rollout_idx][layer_idx][1],
                                rollout_data["input"][rollout_idx][layer_idx][1],
                            )
                            activations[rollout_idx][layer_idx][0] = np.minimum(
                                activations[rollout_idx][layer_idx][0],
                                rollout_data["activation"][rollout_idx][layer_idx][0],
                            )
                            activations[rollout_idx][layer_idx][1] = np.maximum(
                                activations[rollout_idx][layer_idx][1],
                                rollout_data["activation"][rollout_idx][layer_idx][1],
                            )

            # the loss function is between
            # [B, n_roll, state_dim]
            state_pred_err = state_rollout_pred - state_rollout_gt
            loss = 0.0
            # all the losses would be in meters . . . .
            # loss_mse = MSELoss(state_rollout_pred, state_rollout_gt)
            loss_mse = torch.sum(torch.square(state_pred_err)) / torch.sum(torch.ones_like(weights)) / n_roll
            # loss_mse = (torch.mean(torch.square(state_pred_err),dim=(1,2))*weights).sum()/weights.sum()
            loss += loss_mse

            # normalization loss for latent task
            if "latent" in task_name:
                gt_norm = torch.norm(state_rollout_gt, norm_type, dim=-1)
                pred_norm = torch.norm(state_rollout_pred, norm_type, dim=-1)
                loss_container["gt_norm"] = gt_norm.mean()
                loss_container["pred_norm"] = pred_norm.mean()
                if use_norm_reg:
                    loss_norm = lam_norm_reg * torch.mean((gt_norm - pred_norm) ** 2)
                    loss += loss_norm
                    loss_container["norm_loss"] = loss_norm

            # L1 regularization loss
            loss_reg_l1 = 0.0
            loss_reg_l2 = 0.0
            n_param = 0.0
            for ii, W in enumerate(list(model.model.parameters())):
                if ii % 2 == 0:  # only do this for the weights
                    loss_reg_l1 += W.norm(p=1)
                    loss_reg_l2 += W.norm(p=2)
                    n_param += W.numel()  # number of elements in input tensor
            loss_reg = ((0.5 * loss_reg_l1 + 0.5 * loss_reg_l2) / n_param) * float(train_config["lam_l1_reg"])

            loss += loss_reg

            # IBP loss
            if (not is_gnn) and use_ibp_loss:
                loss_ibp = 0.0
                # get lower and upper bounds of the input
                input_lb = np.min([inputs[rollout_idx][0][0] for rollout_idx in range(n_roll)], axis=0)
                input_ub = np.max([inputs[rollout_idx][0][1] for rollout_idx in range(n_roll)], axis=0)
                input_lb = torch.tensor(input_lb, dtype=torch.float32)
                input_ub = torch.tensor(input_ub, dtype=torch.float32)
                if torch.cuda.is_available():
                    input_lb = input_lb.cuda()
                    input_ub = input_ub.cuda()
                # compute IBP bounds
                bounds_per_layer = model.ibp_forward(input_lb, input_ub)
                for lb, ub in bounds_per_layer:
                    loss_ibp += torch.sum(ub - lb) / B
                loss_ibp = loss_ibp * float(config["train"]["lam_ibp_loss"])
                loss += loss_ibp
                loss_container["ibp"] = loss_ibp
                meter_loss_ibp.update(loss_ibp.item(), B)

            meter_loss.update(loss.item(), B)
            meter_loss_rmse.update(np.sqrt(loss_mse.item()), B)
            meter_loss_reg.update(loss_reg.item(), B)

            # compute losses at final step of the rollout
            mse_final_step = MSELoss(state_rollout_pred[:, -1, :], state_rollout_gt[:, -1, :])
            l2_final_step = torch.norm(state_pred_err[:, -1], dim=-1).mean()

            loss_container["mse"] = loss_mse
            loss_container["mse_final_step"] = mse_final_step
            loss_container["l2_final_step"] = l2_final_step
            loss_container["reg"] = loss_reg

        if i % config["log"]["log_per_iter"] == 0 or i == len(data_loader) - 1:  # and i != 0:
            # log_memory_usage()
            log = "Epoch: [%d/%d] Phase: valid Step:[%d/%d]" % (
                epoch + 1,
                epochs,
                i,
                len(data_loader),
            )
            log += ", loss: %.6f (%.6f)" % (loss.item(), meter_loss.avg)
            log += ", rmse: %.6f (%.6f)" % (np.sqrt(loss_mse.item()), meter_loss_rmse.avg)
            log += ", reg: %.6f (%.6f)" % (loss_reg.item(), meter_loss_reg.avg)
            log += ", best_valid_loss: %.6f" % best_valid_loss
            print(log)

            # log data to tensorboard
            log_dict = {}
            
            for loss_type, loss_obj in loss_container.items():
                # plot_name = "Loss/%s/%s" % (loss_type, phase)
                log_dict['valid' + "_" + loss_type] = loss_obj.item()
            if exp is not None:
                exp.write(eval_metrics=log_dict)
    
    if meter_loss.avg < best_valid_loss:
        best_valid_loss = meter_loss.avg
        if exp is not None:
            torch.save(model, f"{exp._save_dir}/best_model.pth")

    return best_valid_loss, inputs, activations


def train_one_epoch(data_loader, model, optimizer, scheduler, exp, config, epoch):
    model.train()
    MSELoss = nn.MSELoss()
    task_name = config["task_name"]
    train_config = config["train"]
    epochs = train_config["n_epoch_initial"]
    num_layers = len(train_config["architecture"])
    robust_training = train_config["robust"]
    data_config = config["data"]
    if robust_training:
        robust_epsilon = train_config["robust_settings"]["epsilon"]
        robust_alpha = train_config["robust_settings"]["alpha"]
        robust_iters = train_config["robust_settings"]["iters"]
    if "latent" in task_name:
        latent_config = config["latent"]
        use_norm_reg = latent_config["use_norm_reg"]
        norm_type = float(latent_config["norm_type"])
        lam_norm_reg = latent_config["lam_norm_reg"]
    use_ibp_loss = config["train"]["use_ibp_loss"]
    step_weight_ub = train_config["step_weight_ub"]
    computing_bounds = train_config.get("compute_bounds", False) and epoch == epochs - 1
    if use_ibp_loss:
        meter_loss_ibp = AverageMeter()
    is_gnn = config["model"] == "PyG_GNN"
    # set up recording metrics
    meter_loss = AverageMeter()
    meter_loss_rmse = AverageMeter()
    meter_loss_reg = AverageMeter()

    inputs = []
    activations = []
    
    for i, data in enumerate(data_loader):
        loss_container = dict()  # store the losses for this step
        with torch.set_grad_enabled(True):
            # n_his, n_roll = train_config["n_history"], train_config["n_rollout"]
            # n_samples = n_his + n_roll # trajectory length
            n_his = data_loader.dataset.n_his
            n_roll = data_loader.dataset.n_roll
            n_samples = data_loader.dataset.n_sample
            # [B, n_samples, obs_dim]
            observations = data["observations"]
            # [B, n_samples, action_dim]
            actions = data["actions"]
            pusher_pos = data["pusher_pos"]
            B = actions.shape[0]
            # [B, n_roll]
            weights = data["weights"]
            step_weight = torch.linspace(1, step_weight_ub, n_roll)
            if torch.cuda.is_available():
                observations = observations.cuda()
                actions = actions.cuda()
                weights = weights.cuda()
                step_weight = step_weight.cuda()
            # [B, n_roll]
            loss_weights = (weights * step_weight).unsqueeze(-1)
            # states, actions = data
            assert actions.shape[1] == n_samples
            loss_mse = 0.0
            # we don't have any visual observations, so states are observations
            states = observations.float() # [B, n_samples, obs_dim]
            # [B, n_his, state_dim]
            state_init = states[:, :n_his, :model.state_dim]
            
            # We want to rollout n_roll steps
            action_seq = actions[:, :-1].float() # [B, n_his+n_roll-1, action_dim]
            input_dict = {"state_init": state_init, "action_seq": action_seq, "pusher_pos": pusher_pos,}
            # [B, n_roll, state_dim]
            state_rollout_gt = states[:, n_his:]
            # try using models_dy.rollout_model instead of doing this manually
            verbose = computing_bounds or use_ibp_loss
            if robust_training:
                state_init_adv = pgd_attack(
                    model, input_dict, state_rollout_gt, robust_epsilon, robust_alpha, robust_iters
                )
                input_dict = {"state_init": state_init_adv, "action_seq": action_seq, "pusher_pos": pusher_pos,}
                rollout_data = model.rollout_model(
                    input_dict=input_dict, verbose=verbose
                )
            else:
                rollout_data = model.rollout_model(
                    input_dict=input_dict, verbose=verbose
                )

            # [B, n_roll, state_dim]
            state_rollout_pred = rollout_data["state_pred"]
            
            if (not is_gnn) and verbose:
                if inputs == []:
                    inputs = rollout_data["input"]
                    activations = rollout_data["activation"]
                else:
                    for rollout_idx in range(n_roll):
                        for layer_idx in range(num_layers):
                            inputs[rollout_idx][layer_idx][0] = np.minimum(
                                inputs[rollout_idx][layer_idx][0],
                                rollout_data["input"][rollout_idx][layer_idx][0],
                            )
                            inputs[rollout_idx][layer_idx][1] = np.maximum(
                                inputs[rollout_idx][layer_idx][1],
                                rollout_data["input"][rollout_idx][layer_idx][1],
                            )
                            activations[rollout_idx][layer_idx][0] = np.minimum(
                                activations[rollout_idx][layer_idx][0],
                                rollout_data["activation"][rollout_idx][layer_idx][0],
                            )
                            activations[rollout_idx][layer_idx][1] = np.maximum(
                                activations[rollout_idx][layer_idx][1],
                                rollout_data["activation"][rollout_idx][layer_idx][1],
                            )

            # the loss function is between
            # [B, n_roll, state_dim]
            state_pred_err = state_rollout_pred - state_rollout_gt
            loss = 0.0
            # all the losses would be in meters . . . .
            # loss_mse = MSELoss(state_rollout_pred, state_rollout_gt)
            loss_mse = torch.sum(loss_weights * torch.square(state_pred_err)) / torch.sum(loss_weights) / n_roll
            # loss_mse = (torch.mean(torch.square(state_pred_err),dim=(1,2))*weights).sum()/weights.sum()
            loss += loss_mse
            # normalization loss for latent task
            if "latent" in task_name:
                gt_norm = torch.norm(state_rollout_gt, norm_type, dim=-1)
                pred_norm = torch.norm(state_rollout_pred, norm_type, dim=-1)
                loss_container["gt_norm"] = gt_norm.mean()
                loss_container["pred_norm"] = pred_norm.mean()
                if use_norm_reg:
                    loss_norm = lam_norm_reg * torch.mean((gt_norm - pred_norm) ** 2)
                    loss += loss_norm
                    loss_container["norm_loss"] = loss_norm

            if "merging_L" == task_name:
                l_shape_loss = l_shape_penalty(state_rollout_pred, state_rollout_gt, unit_size=data_config['unit_size']/data_config['scale'])
                # import pdb; pdb.set_trace()
                l_shape_loss = torch.sum(l_shape_loss.unsqueeze(-1) * loss_weights) / torch.sum(loss_weights) / n_roll * train_config['shape_loss']
                loss += l_shape_loss
                loss_container["shape_loss"] = l_shape_loss
            if "pushing_T" == task_name:
                t_shape_loss = t_shape_penalty(state_rollout_pred, state_rollout_gt, stem_size=np.array(list(data_config['stem_size']))/data_config['scale'], bar_size=np.array(list(data_config['bar_size']))/data_config['scale'])
                t_shape_loss = torch.sum(t_shape_loss.unsqueeze(-1) * loss_weights) / torch.sum(loss_weights) / n_roll * train_config['shape_loss']
                loss += t_shape_loss
                loss_container["shape_loss"] = t_shape_loss

            # interval loss for obj_pile task
            if "obj_pile" in task_name and train_config['interval_loss_ratio'] > 0:
                loss_interval = train_config['interval_loss_ratio'] * soft_interval_loss(state_rollout_pred, interval=data_config['obj_size']*2/data_config['scale'])
                loss += loss_interval
                loss_container["interval"] = loss_interval
            
            if config["task_name"] == "reorientation":
                loss_shape = box_shape_loss(state_rollout_pred)
                loss_container["shape"] = loss_shape
                loss += loss_shape
            # L1 regularization loss
            loss_reg_l1 = 0.0
            loss_reg_l2 = 0.0
            n_param = 0.0
            for ii, W in enumerate(list(model.model.parameters())):
                if ii % 2 == 0:  # only do this for the weights
                    loss_reg_l1 += W.norm(p=1)
                    loss_reg_l2 += W.norm(p=2)
                    n_param += W.numel()  # number of elements in input tensor
            loss_reg = ((0.5 * loss_reg_l1 + 0.5 * loss_reg_l2) / n_param) * float(train_config["lam_l1_reg"])

            loss += loss_reg

            # IBP loss
            if (not is_gnn) and use_ibp_loss:
                loss_ibp = 0.0
                # get lower and upper bounds of the input
                input_lb = np.min([inputs[rollout_idx][0][0] for rollout_idx in range(n_roll)], axis=0)
                input_ub = np.max([inputs[rollout_idx][0][1] for rollout_idx in range(n_roll)], axis=0)
                input_lb = torch.tensor(input_lb, dtype=torch.float32)
                input_ub = torch.tensor(input_ub, dtype=torch.float32)
                if torch.cuda.is_available():
                    input_lb = input_lb.cuda()
                    input_ub = input_ub.cuda()
                # compute IBP bounds
                bounds_per_layer = model.ibp_forward(input_lb, input_ub)
                for lb, ub in bounds_per_layer:
                    loss_ibp += torch.sum(ub - lb) / B
                loss_ibp = loss_ibp * float(config["train"]["lam_ibp_loss"])
                loss += loss_ibp
                loss_container["ibp"] = loss_ibp
                meter_loss_ibp.update(loss_ibp.item(), B)

            meter_loss.update(loss.item(), B)
            meter_loss_rmse.update(np.sqrt(loss_mse.item()), B)
            meter_loss_reg.update(loss_reg.item(), B)

            # compute losses at final step of the rollout
            mse_final_step = MSELoss(state_rollout_pred[:, -1, :], state_rollout_gt[:, -1, :])
            l2_final_step = torch.norm(state_pred_err[:, -1], dim=-1).mean()

            loss_container["mse"] = loss_mse
            loss_container["mse_final_step"] = mse_final_step
            loss_container["l2_final_step"] = l2_final_step
            loss_container["reg"] = loss_reg

        if not computing_bounds: # last epoch is used for computing bounds, so no updates performed
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                if train_config["lr_scheduler"]["type"] == "StepLR":
                    scheduler.step()
                elif train_config["lr_scheduler"]["type"] == "ReduceLROnPlateau":
                    scheduler.step(loss)
                elif train_config["lr_scheduler"]["type"] == "CosineAnnealingLR":
                    scheduler.step()

        if i % config["log"]["log_per_iter"] == 0:  # and i != 0:
            # log_memory_usage()
            log = "Epoch: [%d/%d] Phase: train Step:[%d/%d] LR: %.6f" % (
                epoch + 1,
                epochs,
                i,
                len(data_loader),
                optimizer.param_groups[0]["lr"],
            )
            log += ", loss: %.6f (%.6f)" % (loss.item(), meter_loss.avg)
            log += ", rmse: %.6f (%.6f)" % (np.sqrt(loss_mse.item()), meter_loss_rmse.avg)
            log += ", reg: %.6f (%.6f)" % (loss_reg.item(), meter_loss_reg.avg)
            
            print(log)

            # log data to tensorboard
            log_dict = {}
            
            for loss_type, loss_obj in loss_container.items():
                # plot_name = "Loss/%s/%s" % (loss_type, phase)
                log_dict['train' + "_" + loss_type] = loss_obj.item()
            
            exp.write(train_metrics=log_dict, 
                    **{'lr': optimizer.param_groups[0]["lr"], 'epoch':epoch})

    # Update hard negatives at the end of each training epoch
    if config["data"]["enable_hnm"]:
        hard_negative_indices, errors, info_dict = identify_hard_negatives(
            model, data_loader, config["data"]["hn_th"], step_weight_ub
        )

        # hard_negative_indices = np.vstack((hard_negative_indices//n_roll, hard_negative_indices%n_roll)).T
        new_weight = np.exp(errors * config["data"]["weight_factor"]).clip(1, config["data"]["weight_ub"])
        # new_weight = new_weight.reshape(-1,n_roll)
        data_loader.dataset.update_weights(hard_negative_indices, new_weight)

        del hard_negative_indices, errors, new_weight

        exp.write(info=info_dict)
    
    if epoch != epochs - 1:
        torch.save(model, f"{exp._save_dir}/model_iter_{epoch}.pth")
    else:
        torch.save(model, f"{exp._save_dir}/latest_model.pth")
    
    return inputs, activations

def pgd_attack(model, input_dict, state_rollout_gt, eps, alpha, iters):
    state_init = input_dict['state_init'] # [B, n_his, state_dim]
    state_init_adv = state_init.clone().detach().requires_grad_(True)
    original_state_init = state_init.clone().detach()

    for _ in range(iters):
        rollout_data_adv = model.rollout_model(input_dict=input_dict, grad=True)
        state_rollout_pred_adv = rollout_data_adv["state_pred"]

        loss = nn.MSELoss()(state_rollout_pred_adv, state_rollout_gt)
        model.zero_grad()
        loss.backward()
        state_init_adv = state_init_adv + alpha * state_init_adv.grad.sign()
        eta = torch.clamp(state_init_adv - original_state_init, min=-eps, max=eps)
        # state_init_adv = torch.clamp(original_state_init + eta, min=0, max=1).detach_()
        state_init_adv = (original_state_init + eta).detach_()
        state_init_adv.requires_grad_(True)

    return state_init_adv


def identify_hard_negatives(model, dataloader, value=0.25, step_weight_ub=1):
    model.config["training"] = False
    model.eval()  # Set the model to evaluation mode
    errors = []
    n_his, n_roll = model.config["train"]["n_history"], model.config["train"]["n_rollout"]
    step_weight = torch.linspace(1, step_weight_ub, n_roll)
    step_weight /= step_weight.sum() / n_roll
    with torch.no_grad():
        for data in dataloader:
            observations = data["observations"]
            actions = data["actions"]
            pusher_pos = data["pusher_pos"]
            if torch.cuda.is_available():
                observations = observations.cuda()
                actions = actions.cuda()
                pusher_pos = pusher_pos.cuda()
                
            # Prepare inputs for prediction
            states = observations.float()
            state_init = states[:, :n_his, : model.state_dim]
            
            action_seq = actions[:, :-1].float()
            input_dict = {"state_init": state_init, "action_seq": action_seq, "pusher_pos": pusher_pos,}
            # Predict next states
            rollout_data = model.rollout_model(input_dict=input_dict)
            state_rollout_pred = rollout_data["state_pred"]

            # Ground truth states
            state_rollout_gt = states[:, n_his:]

            # Calculate mean squared error for each sample in the batch
            batch_errors = torch.mean((state_rollout_pred - state_rollout_gt) ** 2, dim=(1, 2))
            errors.append(batch_errors)
    # Concatenate all errors and sort them
    all_errors = torch.cat(errors)
    print("error percentile", [round(np.percentile(all_errors.cpu().numpy(), 5 * i), 5) for i in range(21)])
    threshold = value

    # Get indices of hard negatives
    hard_negative_indices = torch.where(all_errors >= threshold)[0]
    hn_rate = len(hard_negative_indices) / len(all_errors)
    print("hard negative rate: ", hn_rate)
    model.config["training"] = True
    model.train()
    return (
        hard_negative_indices.cpu().numpy(),
        all_errors[hard_negative_indices].cpu().numpy(),
        {
            "hard_negative_rate": hn_rate,
            "max_error": all_errors.max().item(),
            "min_error": all_errors.min().item(),
            "mean_error": all_errors.mean().item(),
            "median_error": all_errors.median().item(),
            "threshold": threshold,
        },
    )


def l_shape_penalty(state_pred, state_gt, unit_size):
    """
    To keep the shape of L
    """
    B, n_rollout, state_dim = state_pred.shape
    if state_dim != 12:
        return 0
    keypoints = state_pred.view(B, n_rollout, state_dim // 2, 2)
    penalty = 0
    for i in range(2):
        p1 = keypoints[:, :, 3*i, :]
        p2 = keypoints[:, :, 3*i+1, :]
        p3 = keypoints[:, :, 3*i+2, :]
        d12 = torch.norm(p1 - p2, dim=-1)
        d23 = torch.norm(p2 - p3, dim=-1)
        d13 = torch.norm(p1 - p3, dim=-1)
        # import pdb; pdb.set_trace()
        # s12, s23 = 1, 2
        s12, s23 = 1.5, 2.5
        s13 = (s12**2 + s23**2)**0.5
        penalty += torch.abs(d12 - s12*unit_size) + torch.abs(d23 - s23*unit_size) + torch.abs(d13 - s13 * unit_size)
    # import pdb; pdb.set_trace()
    # far_penalty = ((abs(state_gt) > 3*unit_size) * abs(state_pred - state_gt)).sum(dim=-1)
    far_penalty = 0
    return penalty + far_penalty

def t_shape_penalty(state_pred, state_gt, stem_size, bar_size):
    B, n_rollout, state_dim = state_pred.shape
    s_w, s_h = stem_size
    b_w, b_h = bar_size
    # state_dim == 8
    keypoints = state_pred.view(B, n_rollout, state_dim // 2, 2)
    penalty = 0
    p1 = keypoints[:, :, 0, :]
    p2 = keypoints[:, :, 1, :]
    p3 = keypoints[:, :, 2, :]
    p4 = keypoints[:, :, 3, :]
    d12 = torch.norm(p1 - p2, dim=-1)
    d13 = torch.norm(p1 - p3, dim=-1)
    d23 = torch.norm(p2 - p3, dim=-1)
    d14 = torch.norm(p1 - p4, dim=-1)
    d24 = torch.norm(p2 - p4, dim=-1)
    d34 = torch.norm(p3 - p4, dim=-1)
    d12_exp = b_w/2
    d23_exp = b_w/2
    d13_exp = b_w
    d24_exp = b_h/2+s_h
    d14_exp = (d12_exp**2 + d24_exp**2)**0.5
    d34_exp = (d23_exp**2 + d24_exp**2)**0.5
    penalty += torch.abs(d12 - d12_exp) + torch.abs(d23 - d23_exp) + torch.abs(d13 - d13_exp) + torch.abs(d24 - d24_exp) + torch.abs(d14 - d14_exp) + torch.abs(d34 - d34_exp)
    return penalty