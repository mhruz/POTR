# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""

import logging
import statistics
import math
import sys
import wandb
from typing import Iterable

import torch
import deformable_potr_plus.util.misc as utils
from deformable_potr_plus.datasets.data_prefetcher import data_prefetcher


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, log_wandb=False):
    model.train()
    criterion.train()
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200
    losses_all = []

    for item_index, (samples, targets) in enumerate(data_loader):
        samples = [item.to(device, dtype=torch.float32) for item in samples]
        targets = [item.to(device) for item in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses_all.append(losses)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if (item_index + 1) % print_freq == 0:
            print(header, "[{0}/{1}]".format(item_index + 1, len(data_loader)), "lr: " + str(optimizer.param_groups[0]["lr"]), "loss: " + str(losses_all[-1].item()))
            logging.info(header + " [{0}/{1}]".format(item_index + 1, len(data_loader)) + " lr: " + str(optimizer.param_groups[0]["lr"]) + " loss: " + str(losses_all[-1].item()))

            if log_wandb:
                wandb.log({**{f'train_{k}': v for k, v in loss_dict.items()},
                           'train_loss': losses_all[-1].item(),
                           'epoch': epoch,
                           'lr': optimizer.param_groups[0]["lr"]})

    converted_losses = [i.item() for i in losses_all]

    print(header, "Averaged stats:", "lr: " + str(optimizer.param_groups[0]["lr"]), "loss: " + str(statistics.mean(converted_losses)))
    logging.info(header + " Averaged stats:" + " lr: " + str(optimizer.param_groups[0]["lr"]) + " loss: " + str(statistics.mean(converted_losses)))

    return {"lr": optimizer.param_groups[0]["lr"], "loss": statistics.mean(converted_losses)}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, print_freq=10, log_wandb=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    l2_pred_error_distances = []
    l2_pred_error_distances_result_averaging = []

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = [item.to(device, dtype=torch.float32) for item in samples]
        targets = [item.to(device) for item in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        l2_pred_error_distances.append(criterion.get_avg_L2_prediction_error_hungarian_matcher_decoding(outputs, targets))
        l2_pred_error_distances_result_averaging.append(criterion.calculate_avg_L2_distance(model.convert_outputs_average_decoding(outputs), torch.stack(targets)))

        metric_logger.update(loss=sum(loss_dict.values()))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    overall_eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    overall_eval_stats["error_distance_matched_decoding"] = statistics.mean(l2_pred_error_distances)
    overall_eval_stats["error_distance_average_decoding"] = statistics.mean(l2_pred_error_distances_result_averaging)

    if log_wandb:
        wandb.log({f'test_{k}': v for k, v in overall_eval_stats.items()})

    print("Averaged eval stats – loss: " + str(overall_eval_stats["loss"]) + ", error distance (decoded via matcher): " + str(statistics.mean(l2_pred_error_distances)) + " mm" + ", error distance (decoded via averagining): " + str(statistics.mean(l2_pred_error_distances_result_averaging)) + " mm")
    logging.info("Averaged eval stats – loss: " + str(overall_eval_stats["loss"]) + ", error distance: " + str(statistics.mean(l2_pred_error_distances)) + " mm" + ", error distance (decoded via averagining): " + str(statistics.mean(l2_pred_error_distances_result_averaging)) + " mm")

    return overall_eval_stats
