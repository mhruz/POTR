# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import statistics
import torch

import potr_base.util.misc as utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: [[dict]],
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, max_norm: float = 0,
                    print_freq=10):

    model.train()
    criterion.train()
    header = 'Epoch: [{}]'.format(epoch)

    losses_all = []

    for item_index, (samples, targets) in enumerate(data_loader):
        samples = [item.to(device, dtype=torch.float32) for item in samples]
        targets = [item.to(device) for item in targets]

        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        losses_all.append(float(loss_dict["loss_coords"].item()))

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if (item_index + 1) % print_freq == 0:
            print(header, "[{0}/{1}]".format(item_index + 1, len(data_loader)), "lr: " + str(optimizer.param_groups[0]["lr"]), "loss: " + str(losses_all[-1]))

    # gather the stats from all processes
    print(header, "Averaged stats:", "lr: " + str(optimizer.param_groups[0]["lr"]), "loss: " + str(statistics.mean(losses_all)))
    return {"lr": optimizer.param_groups[0]["lr"], "loss": statistics.mean(losses_all)}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, print_freq=10):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = [item.unsqueeze(0).expand(3, 224, 224).to(device, dtype=torch.float32) for item in samples]
        targets = [item.to(device) for item in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        metric_logger.update(loss=loss_dict["loss_coords"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats
