
"""
Post-training DIP script
References:
    Croc: https://github.com/naver/croco
    MAE: https://github.com/facebookresearch/mae
    DeiT: https://github.com/facebookresearch/deit
    BEiT: https://github.com/microsoft/unilm/tree/master/beit
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Dict, Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import wandb
import yaml
from termcolor import colored

import utils_dip.misc as misc
from utils_dip.misc import NativeScalerWithGradNormCount as NativeScaler

from datasets import PairsDatasetCOCO, PairsDatasetImagenet
# from datasets.dataset_Imagenet import PairsDataset as PairsDatasetImagenet

from models.dipnet import DIPNet
from models.criterion import CELoss


def get_args_parser() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('DIP pre-training', add_help=False)
    # Model and criterion parameters
    parser.add_argument('--dataset', default='ImageNet_preprocessed_dinoSmall', type=str, help="Training set")
    # Training parameters
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--epochs', default=80, type=int, help="Maximum number of epochs for the scheduler")
    parser.add_argument('--max_epoch', default=40, type=int, help="Stop training at this epoch")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='Learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR', help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='Lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N', help='Epochs to warmup lr')
    parser.add_argument('--amp', type=int, default=1, choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    # Distributed training and logging parameters
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    parser.add_argument('--save_freq', default=1, type=int, help='Frequency (in epochs) to save checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=3, type=int, help='Frequency (in epochs) to save checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=1, type=int, help='Frequency (in iterations) to print info while training')
    # Paths and configuration
    parser.add_argument('--output_dir', default='./output/', type=str, help="Path where to save the output")
    parser.add_argument('--data_dir_base', default='./data/', type=str, help="Path where data are stored")
    parser.add_argument('--config', type=str, required=True, help="Path to model config")
    return parser


def load_model_config(file: str) -> Dict[str, Any]:
    """Load model configuration from a YAML file."""
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def update_args_with_config(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Override arguments with any corresponding entries in the config."""
    for key in ["max_epoch", "epochs", "warmup_epochs", "lr", "save_freq", "keep_freq", "dataset"]:
        if key in config:
            setattr(args, key, config[key])


def setup_environment(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Set up distributed mode, random seeds, and logging."""
    misc.init_distributed_mode(args)
    # Initialize WandB only on the main process.
    if misc.get_rank() == 0:
        args_dict = {**vars(args), **config}
        wandb.init(project="dip-pretraining", config=args_dict, resume="allow", name=config.get('exp_name', 'exp'))
    # Set device and random seed.
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True


def build_output_dir(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    """Build and return the output directory."""
    exp_name = config.get('exp_name', 'default_exp')
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir


def build_datasets(args: argparse.Namespace, config: Dict[str, Any]) -> (Iterable, Iterable):
    """Build training and validation datasets and return DataLoaders."""
    data_params = config.get("data_params", {})
    pairs_file_directory = config.get("pairs_file_directory", '/home/ssirkoga/workspace/crocodino/data/')
    dataset_val_name = config.get("dataset_val", args.dataset)
    print(f"Building dataset for {args.dataset} with parameters: {data_params}")

    if "COCO" in args.dataset:
        dataset_train = PairsDatasetCOCO(args.dataset, data_dir_base=config["data_dir_base"],
                                                split='train', pairs_file_directory=pairs_file_directory, **data_params)
        dataset_val = PairsDatasetCOCO(dataset_val_name, data_dir_base=config["data_dir_base"],
                                              split='val', pairs_file_directory=pairs_file_directory, **data_params)
    elif "ImageNet" in args.dataset:
        dataset_train = PairsDatasetImagenet(args.dataset, data_dir_base=config["data_dir_base"],
                                                    split='train', pairs_file_directory=pairs_file_directory, **data_params)
        dataset_val = PairsDatasetImagenet(dataset_val_name, data_dir_base=config["data_dir_base"],
                                                  split='val', pairs_file_directory=pairs_file_directory, **data_params)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    world_size = misc.get_world_size()
    global_rank = misc.get_rank()

    if world_size > 1:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=world_size, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=world_size, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=config['batch_size'],
        num_workers=config["num_workers"], pin_memory=True, drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=2,
        num_workers=config["num_workers"], pin_memory=True, drop_last=True
    )

    return data_loader_train, data_loader_val


def build_model_and_optimizer(args: argparse.Namespace, config: Dict[str, Any], device: torch.device):
    """Initialize the model, criterion, and optimizer."""
    model_params = config["model_params"]
    model_name = config["model_name"]
    print(f"Loading model: {model_name} with parameters: {model_params}")
    model = eval(model_name)(**model_params)
    criterion = CELoss()
    model.to(device)
    model_without_ddp = model

    # Set up effective batch size and learning rate.
    world_size = misc.get_world_size()
    args.accum_iter = 384 // config['batch_size'] // world_size
    blr = config.get("base_lr", args.blr)
    eff_batch_size = config['batch_size'] * args.accum_iter * world_size
    if args.lr is None:
        args.lr = blr * eff_batch_size / 256

    print(f"Base lr: {args.lr * 256 / eff_batch_size:.2e}, actual lr: {args.lr:.2e}")
    print(f"Accumulate grad iterations: {args.accum_iter}, effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    # Setup optimizer with parameter groups.
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    for group in param_groups:
        group['params'] = [p for p in group['params'] if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print("Optimizer:", optimizer)

    loss_scaler = NativeScaler()
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    return model, model_without_ddp, criterion, optimizer, loss_scaler


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                    loss_scaler, args: argparse.Namespace, log_writer: SummaryWriter = None) -> Dict[str, float]:
    """Training loop for one epoch."""
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print(f"Logging to: {log_writer.log_dir}")

    for data_iter_step, (image1, image2, labels1, labels2) in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)):
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        labels1 = labels1.to(device, non_blocking=True)
        labels2 = labels2.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=bool(args.amp)):
            output = model(image1, image2, labels2)
            loss = criterion(output, labels1.squeeze())
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Rank {args.rank} Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=((data_iter_step + 1) % accum_iter == 0))
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            wandb.log({"train_loss": loss_value_reduce, "lr": lr, "epoch": epoch})

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
                  device: torch.device, epoch: int, args: argparse.Namespace,
                  log_writer: SummaryWriter = None) -> Dict[str, float]:
    """Validation loop for one epoch."""
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'

    with torch.no_grad():
        for data_iter_step, (image1, image2, labels1, labels2) in enumerate(
                metric_logger.log_every(data_loader, args.print_freq, header)):
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            labels1 = labels1.to(device, non_blocking=True)
            labels2 = labels2.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                output = model(image1, image2, labels2)
                loss = criterion(output, labels1.squeeze())
            loss_value = loss

            _, predicted = torch.max(output, 1)
            labels1 = labels1.squeeze()
            correct_pixels = (predicted == labels1).sum().item()
            total_pixels = labels1.numel()
            accuracy = correct_pixels / total_pixels

            metric_logger.update(val_accuracy=accuracy)
            if not math.isfinite(loss_value):
                print(f"Rank {args.rank} Loss is {loss_value}, stopping training")
                sys.exit(1)

            torch.cuda.synchronize()
            metric_logger.update(loss_val=loss_value)
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and ((data_iter_step + 1) % args.print_freq) == 0:
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('val_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('val_accuracy', accuracy, epoch_1000x)

    metric_logger.synchronize_between_processes()
    average_accuracy = metric_logger.meters["val_accuracy"].global_avg
    print("Mean Validation Accuracy:", average_accuracy)
    if log_writer is not None:
        wandb.log({"val_loss": metric_logger.meters["loss_val"].global_avg, "val_accuracy": average_accuracy, "epoch": epoch})
        print("Validation loss:", metric_logger.meters["loss_val"].global_avg)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args: argparse.Namespace, config: Dict[str, Any]):
    update_args_with_config(args, config)
    setup_environment(args, config)
    output_dir = build_output_dir(args, config)

    # Auto resume if checkpoint exists.
    last_ckpt_fname = output_dir / 'checkpoint-last.pth'
    args.resume = str(last_ckpt_fname) if last_ckpt_fname.is_file() else None

    print(f"Job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"Args:\n{json.dumps(vars(args), indent=2)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader_train, data_loader_val = build_datasets(args, config)

    model, model_without_ddp, criterion, optimizer, loss_scaler = build_model_and_optimizer(args, config, device)

    # Calculate effective batch size and update learning rate if needed.
    world_size = misc.get_world_size()
    args.accum_iter = 384 // config['batch_size'] // world_size
    eff_batch_size = config['batch_size'] * args.accum_iter * world_size

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    # Build optimizer parameter groups for weight decay rules.
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    for group in param_groups:
        group['params'] = [p for p in group['params'] if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print("Optimizer:", optimizer)

    # Load previous checkpoint if available.
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    log_writer = SummaryWriter(log_dir=str(output_dir)) if misc.is_main_process() else None

    print(f"Start training until {args.max_epoch} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.max_epoch):
        if world_size > 1:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer,
                                      device, epoch, loss_scaler, args=args, log_writer=log_writer)
        val_stats = val_one_epoch(model, criterion, data_loader_val, device, epoch, args=args, log_writer=log_writer)

        # Save checkpoint at specified intervals.
        if output_dir and epoch % args.save_freq == 0:
            misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, fname='last', output_dir=str(output_dir))
        if output_dir and (epoch % args.keep_freq == 0 or epoch + 1 == args.max_epoch) and (epoch > 0 or args.max_epoch == 1):
            misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, output_dir=str(output_dir), epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}
        if output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(output_dir / "log.txt", mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time: {total_time_str}')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    print(colored("Loading configuration...", "light_blue"))
    config = load_model_config(args.config)
    main(args, config)
