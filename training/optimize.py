import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from training.dataset import DynamicsDataset, DynamicsDataset_IMG
import gc
import wandb
from training.dataset import DynamicsDataset
from training.flex_dataset import FlexDynamicsDataset
from training.flex_dataset_rope_3d import FlexRope3DDynamicsDataset
from training.sapien_dataset_reorientation import ReorientationDynamicsDataset
from training.engines import train_one_epoch, validate
from training.sampler import sample_data, online_traj_sample
import pickle

def optimize(
    config,  # yaml config
    model,  # model object
    exp,  # exp logger
):
    datasets = {}
    dataloaders = {}
    data_n_batches = {}
    train_config = config["train"]
    task_name = config["task_name"]
    DatasetClass = DynamicsDataset
    if "latent" in task_name:
        DatasetClass = DynamicsDataset_IMG
    if config['task_name'] == 'pushing_rope':
        DatasetClass = FlexDynamicsDataset
    elif config['task_name'] == 'obj_pile':
        DatasetClass = DynamicsDataset
    elif config['task_name'] == 'rope_3d':
        DatasetClass = FlexRope3DDynamicsDataset
    elif config['task_name'] == 'reorientation':
        DatasetClass = ReorientationDynamicsDataset
    print(f"DatasetClass: {DatasetClass}")
    
    for phase in ["train", "valid"]:
        datasets[phase] = DatasetClass(config, phase=phase)
        dataloaders[phase] = DataLoader(
            datasets[phase],
            batch_size=train_config["batch_size"],
            # shuffle=False,
            pin_memory=True,
            shuffle=True if phase == 'train' and config["data"]["enable_hnm"] == False else False,
            num_workers=train_config["num_workers"],
            drop_last=True,
        )
        data_n_batches[phase] = len(dataloaders[phase])

    # optimizer
    params = model.parameters()
    lr = float(train_config["lr"])
    optimizer = optim.Adam(params, lr=lr, betas=(train_config["lr_params"]["adam_beta1"], 0.999))
    epochs = train_config["n_epoch_initial"]
    # setup scheduler
    sc_config = train_config["lr_scheduler"]
    scheduler = None
    if train_config["lr_scheduler"]["enabled"]:
        if train_config["lr_scheduler"]["type"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=sc_config["factor"],
                patience=sc_config["patience"],
                threshold=sc_config["threshold"],
                threshold_mode=sc_config["threshold_mode"],
                cooldown=sc_config["cooldown"],
                min_lr=sc_config["min_lr"],
                eps=sc_config["eps"],
                verbose=True,
            )
        elif train_config["lr_scheduler"]["type"] == "StepLR":
            step_size = train_config["lr_scheduler"]["step_size"]
            gamma = train_config["lr_scheduler"]["gamma"]
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif train_config["lr_scheduler"]["type"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, epochs*len(dataloaders['train']), eta_min=0, last_epoch=-1)
        else:
            raise ValueError("unknown scheduler type: %s" % (train_config["lr_scheduler"]["type"]))
    if torch.cuda.is_available():
        print("using gpu")
        model = model.cuda()
    
    computing_bounds = train_config.get("compute_bounds", False)
    best_valid_loss = np.inf

    inputs = []
    activations = []    
    if train_config["online_training"]["enable"]:
        online_dataset = []
    for epoch in range(epochs):        
        train_inputs, train_activations = train_one_epoch(dataloaders['train'], model, optimizer, scheduler, exp, config, epoch)
        print(f"train finished epoch {epoch}")
        best_valid_loss, valid_inputs, valid_activations = validate(dataloaders['valid'], model, exp, config, epoch, best_valid_loss)
        if train_config["online_training"]["enable"] and epoch % train_config["online_training"]["sample_data_per_epoch"] == 0: # and epoch != 0
            # import pdb; pdb.set_trace()
            assert config["data"]["episode_length"] == dataloaders['train'].dataset.episode_length
            additional_dataset = online_traj_sample(config, model, num_episodes=train_config["online_training"]["num_episodes"], \
                num_processes=2, episode_length=config["data"]["episode_length"])
            online_dataset.extend(additional_dataset)
            dataloaders['train'].dataset.add_online_data(additional_dataset)
        
        if computing_bounds and epoch == epochs - 1:
            inputs.extend(train_inputs)
            inputs.extend(valid_inputs)
            activations.extend(train_activations)
            activations.extend(valid_activations)
        
    if train_config["online_training"]["enable"]:
        online_dataset = np.array(online_dataset)
        with open(f"{exp._save_dir}/online_data.p", "wb") as fp:
            pickle.dump(online_dataset, fp)
        print(f"save {exp._save_dir}/online_data.p")
    
    del datasets, dataloaders
    
    return model, inputs, activations

