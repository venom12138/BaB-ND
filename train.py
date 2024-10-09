import os
import random
import sys

import hydra
import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf
import warnings
warnings.filterwarnings("ignore")
import wandb
from model.mlp import MLP
from model.model import Rope_MLP, PyG_GNN, Reori_MLP
from training.optimize import optimize
from util.exp_handler import ExpHandler

import torch.multiprocessing as mp
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# default_config does not exist, just a placeholder
@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(config: DictConfig) -> None:
    if config.get("device_id", False):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_id"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    exp = ExpHandler(config=config, en_wandb=config.wandb)

    is_gnn = config["model"] == "PyG_GNN"
    if not is_gnn and config["train"]["include_com"]:
        config["data"]["state_dim"] += 2 * config["data"]["obj_num"]
    # create the dynamics model
    model = eval(config["model"])(config)
    checkpoint_path = config.train.get("checkpoint_path", None)
    if checkpoint_path is not None:
        model = torch.load(checkpoint_path).eval()
        model.config = config
    model = model.cuda()
    print(f"model params: {sum(p.numel() for p in model.parameters())}")

    # model.load_state_dict(torch.load(config['model_path']))
    # online_traj_sample(config, model, num_episodes=2, num_processes=18, \
    #     episode_length=12, verbose=True)
    
    ### optimize the dynamics model
    model, model_input, activation = optimize(
        config,
        model,
        exp,
    )

    if not is_gnn:
        model.update_bounds(model_input, activation)
    del model_input, activation

    torch.save(model, f'{exp._save_dir}/latest_model.pth')
    exp.finish()
    print("training id: ", exp._exp_id)
    print("model saved at: ", f'{exp._save_dir}/latest_model.pth')
if __name__ == "__main__":
    main()
