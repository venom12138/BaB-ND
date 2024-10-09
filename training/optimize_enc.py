import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from model.encoder import AE
from training.dataset import ImageDataset, ContrastiveImageDataset
from others.meter import AverageMeter
def distance(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)

def optimize(model: AE, config, exp):
    datasets = {}
    dataloaders = {}
    data_n_batches = {}
    latent_config = config["latent"]
    if not config['latent']['contrastive']:
        DatasetClass = ImageDataset
    else:
        DatasetClass = ContrastiveImageDataset
    for phase in ["valid", "train"]:
        datasets[phase] = DatasetClass(config, phase=phase)
        dataloaders[phase] = DataLoader(
            datasets[phase],
            batch_size=latent_config["batch_size"],
            # shuffle=False,
            pin_memory=True,
            shuffle=True if phase == 'train' and latent_config["enable_hnm"] == False else False,
            num_workers=latent_config["num_workers"],
            drop_last=True,
        )
        data_n_batches[phase] = len(dataloaders[phase])

    best_valid_loss = np.inf
    lam_l1_reg = latent_config["lam_l1_reg"]
    use_norm_reg = latent_config["use_norm_reg"]
    norm_type = float(latent_config["norm_type"])
    lam_norm_reg = latent_config["lam_norm_reg"]

    # optimizer
    params = model.parameters()
    lr = float(latent_config["lr"])
    optimizer = optim.AdamW(params, lr=lr, betas=(latent_config["lr_params"]["adam_beta1"], 0.995), eps=1e-8)

    # setup scheduler
    sc = latent_config["lr_scheduler"]
    scheduler = None
    epochs = latent_config["n_epoch"]
    if latent_config["lr_scheduler"]["enabled"]:
        if latent_config["lr_scheduler"]["type"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=sc["factor"],
                patience=sc["patience"],
                threshold=sc["threshold"],
                threshold_mode=sc["threshold_mode"],
                cooldown=sc["cooldown"],
                min_lr=sc["min_lr"],
                eps=sc["eps"],
                verbose=True,
            )
        elif latent_config["lr_scheduler"]["type"] == "StepLR":
            step_size = latent_config["lr_scheduler"]["step_size"]
            gamma = latent_config["lr_scheduler"]["gamma"]
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif latent_config["lr_scheduler"]["type"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, epochs*len(dataloaders['train']), eta_min=0, last_epoch=-1)
        else:
            raise ValueError("unknown scheduler type: %s" % (latent_config["lr_scheduler"]["type"]))

    if torch.cuda.is_available():
        print("using gpu")
        model = model.cuda()

    
    phases = ["train", "valid"]
    global_iteration = 0
    for epoch in range(epochs):
        for phase in phases:
            # set up recording metrics
            meter_loss = AverageMeter()
            meter_loss_rmse = AverageMeter()
            meter_loss_reg = AverageMeter()
            meter_loss_norm = AverageMeter()

            loader = dataloaders[phase]
            for i, data in enumerate(loader):
                loss_container = dict()
                global_iteration += 1
                with torch.set_grad_enabled(phase == "train"):
                    inputs = data["observations"].float()
                    explicit_state = data["explicit_state"].float()
                    B, C, H, W = inputs.shape
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        explicit_state = explicit_state.cuda()
                        if config['latent']['contrastive']:
                            pos_sample = data["positive"].float().cuda()
                            neg_sample = data["negative"].float().cuda()
                        elif config['latent']['latent_reg']:
                            pos_sample = data["positive"].float().cuda()
                    loss = 0.
                        
                    if latent_config['model'] == 'VAE':
                        outputs, latent = model(inputs)
                        # import pdb; pdb.set_trace()
                        logvar = model.encoder.logvar
                        mu = model.encoder.mu
                        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                        loss += 0.00025 * KLD
                        loss_container['KL_loss'] = 0.00025 * KLD.item()
                        
                    elif latent_config['model'] == 'AE':
                        outputs, latent = model(inputs)
                        if config['latent']['contrastive']:
                            pos_output, pos_latent = model(pos_sample)
                            neg_output, neg_latent = model(neg_sample)
                            d_positive = distance(latent, pos_latent)
                            d_negative = distance(latent, neg_latent)
                            loss_cons = torch.clamp(2.0 + d_positive - d_negative, min=0.0).mean()
                            loss += loss_cons
                            loss_container["cons"] = loss_cons.item()
                            outputs = torch.cat([outputs, pos_output, neg_output], dim=0)
                            latent = torch.cat([latent, pos_latent, neg_latent], dim=0)
                            inputs = torch.cat([inputs, pos_sample, neg_sample], dim=0)
                        if config['latent']['latent_reg'] > 0:
                            pos_output, pos_latent = model(pos_sample)
                            d_positive = distance(latent, pos_latent)
                            loss_latent_reg = torch.clamp(0.3 - d_positive, min=0.0).mean()
                            loss += loss_latent_reg
                            loss_container["latent_reg"] = loss_latent_reg.item()
                            outputs = torch.cat([outputs, pos_output, ], dim=0)
                            latent = torch.cat([latent, pos_latent, ], dim=0)
                            inputs = torch.cat([inputs, pos_sample, ], dim=0)
                        latent_norm = torch.norm(latent, norm_type, dim=1)
                        loss_container["latent_norm"] = latent_norm.mean().item()
                        if use_norm_reg:
                            loss_norm = lam_norm_reg * torch.mean((latent_norm - 1) ** 2)
                            loss_container["norm_loss"] = loss_norm.item()
                            meter_loss_norm.update(loss_norm.item(), B)
                            loss = loss + loss_norm
                        # use explicit state as supervision
                        # aux_loss = F.mse_loss(model.aux_vector, explicit_state) * 0.1
                        # loss_container["aux"] = aux_loss.item()
                        # loss += aux_loss
                    
                    # loss
                    non_zero_mask = inputs > 0
                    # import pdb; pdb.set_trace()
                    loss_mse = F.mse_loss(inputs, outputs) + 0.1 * F.mse_loss(inputs[non_zero_mask], outputs[non_zero_mask])
                    loss_container["mse"] = loss_mse.item()
                    meter_loss_rmse.update(np.sqrt(loss_mse.item()), B)
                    loss += loss_mse
                    
                    # latent norm regularization loss, oncourge the norm of latent to be 1
                    
                    # L1 regularization
                    l1_reg = torch.tensor(0., requires_grad=True)
                    for name, param in model.named_parameters():
                        if 'weight' in name:
                            l1_reg = l1_reg + torch.norm(param, 1)
                    loss_reg = lam_l1_reg * l1_reg
                    loss = loss + loss_reg
                    
                    loss_container["loss"] = loss.item()
                    meter_loss.update(loss.item(), B)
                    loss_container["reg"] = loss_reg.item()
                    meter_loss_reg.update(loss_reg.item(), B)
                    
                    if phase == "train":
                        optimizer.zero_grad()
                        loss = loss * 10
                        loss.backward()
                        optimizer.step()

                if i % config["log"]["log_per_iter"] == 0:  # and i != 0:
                    log = "Epoch: [%d/%d] Phase: %s Step:[%d/%d] Global Iter: %d LR: %.6f" % (
                        epoch + 1,
                        epochs,
                        phase,
                        i,
                        data_n_batches[phase],
                        global_iteration,
                        optimizer.param_groups[0]["lr"],
                    )
                    log += ", loss: %.6f (%.6f)" % (loss.item(), meter_loss.avg)
                    log += ", rmse: %.6f (%.6f)" % (np.sqrt(loss.item()), meter_loss_rmse.avg)
                    log += ", reg: %.6f (%.6f)" % (loss_reg.item(), meter_loss_reg.avg)
                    if latent_config['model'] == 'AE':
                        log += ", norm: %.6f (%.6f)" % (loss_norm.item(), meter_loss_norm.avg)
                    print(log)
                    log_dict = {}
                    log_dict["lr"] = optimizer.param_groups[0]["lr"]
                    for loss_type, loss_obj in loss_container.items():
                        log_dict[phase + "_" + loss_type] = loss_obj
                    exp.write(train_metrics=log_dict, 
                            **{'lr': optimizer.param_groups[0]["lr"], 'epoch':epoch})

            if phase == "train":
                if scheduler is not None:
                    if latent_config["lr_scheduler"]["type"] == "StepLR":
                        scheduler.step()
                    elif latent_config["lr_scheduler"]["type"] == "ReduceLROnPlateau":
                        scheduler.step(loss)
                    elif latent_config["lr_scheduler"]["type"] == "CosineAnnealingLR":
                        scheduler.step()
            if phase == "valid":
                if meter_loss.avg < best_valid_loss:
                    best_valid_loss = meter_loss.avg
                    model.save_model(exp._save_dir)

    return model