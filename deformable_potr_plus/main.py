# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import logging
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import deformable_potr_plus.util.misc as utils
import deformable_potr_plus.datasets.samplers as samplers
from deformable_potr_plus.engine import evaluate, train_one_epoch
from deformable_potr_plus.models import build_model
from dataset import HPOESAdvancedDataset, HPOESOberwegerDataset, HPOESSequentialDataset
from dataset import augmentation


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable POTR Module', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--save_epoch', default=5, type=int, help='interval of saving the model (in epochs)')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Parametrs: Deformable DETR Variants
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Parameters: Model

    #   -> Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    #   -> Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_classes', default=14, type=int,
                        help="Number of classes for the joints to be labeled as (do not include the background class)")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots (number of landmarks to be regressed)")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # Parameters: Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    #   -> Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    #   -> Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # Parameters: Dataset
    parser.add_argument('--train_data_path', default="/storage/plzen4-ntis/projects/cv/hpoes2/data/NYU/train_comrefV2V_3Dproj.h5",
                        type=str, help="Path to the training dataset H5 file.")
    parser.add_argument('--eval_data_path', default="/storage/plzen4-ntis/projects/cv/hpoes2/data/NYU/test_1_comrefV2V_3Dproj.h5",
                        type=str, help="Path to the evaluation dataset H5 file.")

    parser.add_argument('--output_dir', default='train_test_0', help="Path for saving of the resulting weights and overall model")
    parser.add_argument('--device', default='cuda', help="Device to be used for training and testing")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', default=True, action='store_true', help="Determines whether to perform evaluation on each epoch.")
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--p_augment', default=0.5, type=float, help="Probability of applying augmentation.")
    parser.add_argument('--encoded', default=0, type=int,
                        help="Whether to read the encoded data (=1) or the decoded data (=0) into memory.")
    parser.add_argument('--sequence_length', default=0, type=int, help='Number of video frames to process (0 or 3)')
    return parser


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def main(args):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("train_test_0" + ".log")
        ]
    )

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", n_parameters)

    # Load HPOES data from the same source
    # dataset_train = HPOESAdvancedDataset(args.train_data_path)
    # if args.eval:
    #     dataset_eval = HPOESAdvancedDataset(args.eval_data_path)

    if args.sequence_length == 0:
        dataset_train = HPOESOberwegerDataset(args.train_data_path, transform=augmentation(p_apply=args.p_augment),
                                              encoded=args.encoded, p_augment_3d=args.p_augment)
        #dataset_train = HPOESAdvancedDataset(args.train_data_path, transform=None)
    else:
        dataset_train = HPOESSequentialDataset(args.train_data_path, sequence_length=args.sequence_length,
                                               transform=args.p_augment, encoded=args.encoded)
    if args.eval:
        if args.sequence_length == 0:
            dataset_eval = HPOESOberwegerDataset(args.eval_data_path, encoded=args.encoded)
            #dataset_eval = HPOESAdvancedDataset(args.train_data_path, transform=None)
        else:
            dataset_eval = HPOESSequentialDataset(args.eval_data_path, sequence_length=args.sequence_length,
                                                  encoded=args.encoded)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            if args.eval:
                sampler_eval = samplers.NodeDistributedSampler(dataset_eval, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            if args.eval:
                sampler_eval = samplers.DistributedSampler(dataset_eval, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        if args.eval:
            sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers,
                                   pin_memory=True)
    if args.eval:
        data_loader_eval = DataLoader(dataset_eval, args.batch_size, sampler=sampler_eval, drop_last=False,
                                      num_workers=args.num_workers, pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]

    #for n, p in model_without_ddp.named_parameters():
    #    print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    output_dir = Path(args.output_dir)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

        if args.eval:
            test_stats = evaluate(model, criterion, data_loader_eval, device)

        return

    best_train_loss = None
    best_val_loss = None

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch)
        lr_scheduler.step()

        if args.eval:
            test_stats = evaluate(model, criterion, data_loader_eval, device)

        if args.output_dir:
            checkpoint_paths = [os.path.join(output_dir, 'checkpoint.pth')]
            # extra checkpoint before LR drop and every N epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_epoch == 0:
                checkpoint_paths.append(os.path.join(output_dir, f'checkpoint{epoch:04}.pth'))
            if best_train_loss is None or train_stats["loss"] < best_train_loss:
                checkpoint_paths.append(os.path.join(output_dir, 'checkpoint_best_train_loss.pth'))
                best_train_loss = train_stats["loss"]
            if args.eval:
                if best_val_loss is None or test_stats["loss"] < best_val_loss:
                    checkpoint_paths.append(os.path.join(output_dir, 'checkpoint_best_val_loss.pth'))
                    best_val_loss = test_stats["loss"]

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.eval:
            log_stats.update({f'test_{k}': v for k, v in test_stats.items()})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
