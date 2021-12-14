# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import potr_base.util.misc as utils
from potr_base.engine import evaluate, train_one_epoch
from potr_base.models import build_model
from dataset import HPOESOberwegerDataset, HPOESSequentialDataset
from dataset import augmentation


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--save_epoch', default=5, type=int, help='interval of saving the model (in epochs)')
    parser.add_argument('--clip_max_norm', default=0, type=float, help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=21, type=int,
                        help="Number of query slots (number of landmarks to be regressed)")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)", default=False)
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # Parameters: Dataset
    parser.add_argument('--train_data_path',
                        default="/Users/matyasbohacek/Documents/Academics/Datasets/Custom/FAV_Train_Sample/train_image.h5",
                        type=str, help="Path to the training dataset H5 file.")
    parser.add_argument('--eval_data_path',
                        default="/Users/matyasbohacek/Documents/Academics/Datasets/Custom/FAV_Train_Sample/train_image.h5",
                        type=str, help="Path to the evaluation dataset H5 file.")

    parser.add_argument('--output_dir', default='', help="Path for saving of the resulting weights and overall model")
    parser.add_argument('--device', default="cpu", help="Device to be used for training and testing")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help="HTTP address or full directory of the model to resume training on (ignored if empty)")
    parser.add_argument('--init_weights', default='', help='Init network with custom weights (path to weights)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help="Starting epoch index")
    parser.add_argument('--eval', default=False, action='store_true',
                        help="Determines whether to apply evaluation on each epoch.")
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--print_frequency', default=10, type=int,
                        help="Print frequency of the stats during training and evaluation.")
    parser.add_argument('--p_augment', default=0.5, type=float, help="Probability of applying augmentation.")
    parser.add_argument('--encoded', default=0, type=int,
                        help="Whether to read the encoded data (=1) or the decoded data (=0) into memory.")

    # Parameters: Distributed training
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='URL to be used for setup of distributed training')

    parser.add_argument('--sequence_length', default=0, type=int, help='Number of video frames to process (0 or 3)')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
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

    if args.distributed and args.device == "cuda":
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Divide parameters of the model to specify LR for backbone and the rest of the parameters
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Load HPOES data from the same source
    # dataset_train = HPOESAdvancedDataset(args.train_data_path)
    # if args.eval:
    #     dataset_eval = HPOESAdvancedDataset(args.eval_data_path)

    # dataset_train = HPOESOberwegerDataset(args.train_data_path, transform=augmentation(p_apply=args.p_augment),
    #                                       encoded=args.encoded)

    if args.sequence_length == 0:
        dataset_train = HPOESOberwegerDataset(args.train_data_path, transform=augmentation(p_apply=args.p_augment),
                                                                                    encoded=args.encoded, mode='train')
    else:
        dataset_train = HPOESSequentialDataset(args.train_data_path, sequence_length=args.sequence_length,
                                               transform=args.p_augment, encoded=args.encoded, mode='train')
    if args.eval:
        if args.sequence_length == 0:
            dataset_eval = HPOESOberwegerDataset(args.eval_data_path, encoded=args.encoded, mode='eval')
        else:
            dataset_eval = HPOESSequentialDataset(args.eval_data_path, sequence_length=args.sequence_length,
                                                  encoded=args.encoded, mode='eval')

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        if args.eval:
            sampler_eval = DistributedSampler(dataset_eval, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        if args.eval:
            sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    if args.eval:
        data_loader_eval = DataLoader(dataset_eval, args.batch_size, sampler=sampler_eval,
                                      drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location=args.device)
        model_without_ddp.potr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    if args.init_weights:
        checkpoint = torch.load(args.init_weights, map_location=device)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location=args.device, check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location=args.device)
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # Remove previously saved log
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("w") as f:
            f.write("")

    best_train_loss = None
    best_val_loss = None

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm,
                                      print_freq=args.print_frequency)
        lr_scheduler.step()

        if args.eval:
            test_stats = evaluate(model, criterion, data_loader_eval, device, print_freq=args.print_frequency)

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
            for k, v in test_stats.items():
                log_stats["test_" + k] = v

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('POTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
