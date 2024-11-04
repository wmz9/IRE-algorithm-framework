import os
import datetime
import argparse
import warnings

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode, v2 as transforms

from accelerate import Accelerator
from torch.optim import AdamW, SGD
from ire import IRE_PARAMS
import model_zoo


def training_function(config, args):
    if args.disable_log:
        accelerator = Accelerator(log_with=None, project_dir=args.project_dir)
    else:
        if args.project_name is not None:
            project_name = args.project_name
        else:
            project_name = "vit"
            if args.optim == 'Lion':
                project_name += "_lion"
            if args.no_aug:
                project_name += "_no_aug"
            if args.enable_ire:
                project_name += "_ire"
        accelerator = Accelerator(log_with="wandb", project_dir=args.project_dir)
        accelerator.init_trackers(
            project_name=project_name,
            init_kwargs={"wandb": {"name": args.wandb_run_name}},
        )
    accelerator.print(accelerator.distributed_type)
    # IRE disable accumulate
    if not args.enable_ire:
        accelerator.gradient_accumulation_steps = - \
           (args.expected_batch_size // -(args.batch_size * accelerator.num_processes))
    args.lr *= (args.batch_size * accelerator.num_processes *
                accelerator.gradient_accumulation_steps) / args.expected_batch_size

    # We need to initialize the trackers we use, and also store our configuration
    run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers(run, config)

    # Set the seed before splitting the data.
    if args.seed is not None:
        np.random.seed(args.seed)
        model_zoo.rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    extra_args = {}
    if args.no_aug:
        if args.arch.startswith('ViT'):
            extra_args |= {'dropout': 0.1, 'attention_dropout' :0.1}
        if args.arch.startswith('t') and 'ViT' in args.arch:
            extra_args |= {'drop_rate': 0.1, 'proj_drop_rate': 0.1, 'attn_drop_rate': 0.1, 'drop_path_rate':0.1,} 
    model = model_zoo.model_dict[args.arch](**extra_args)

    if args.optim == 'Lion':
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    if args.enable_ire:
        optimizer = IRE_PARAMS(model.parameters(), optimizer, rank=args.ire_rank, rank_increase=args.rank_increase, prog=args.prog, prog_decay=args.prog_decay)

    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_size = model.image_size  if hasattr(model, 'image_size') else 224
    train_transform = [
            transforms.RandomResizedCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
    ]
    if not args.no_aug:
        train_transform.append(transforms.RandAugment(num_ops=2, magnitude=15, interpolation=InterpolationMode.BILINEAR))
    train_transform += [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            normalize,
    ]
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(train_transform)
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((128, 128) if image_size <= 128 else (256, 256)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        multiprocessing_context='spawn'
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        multiprocessing_context='spawn'
    )

    if not args.no_aug:
        mixup = transforms.MixUp(alpha=0.5, num_classes=1000)

    # cosine decay with 30 epochs warmup
    step_per_epoch = -(len(train_dataset) // -(args.batch_size * accelerator.gradient_accumulation_steps))
    total_steps = args.epochs * step_per_epoch
    warmup_steps = args.warmup_epochs * step_per_epoch
    scheduler_1 = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1 / warmup_steps, total_iters=warmup_steps)
    scheduler_2 = lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=total_steps - warmup_steps)  # with '- warmup_steps' optionally
    scheduler = lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[
                                          scheduler_1, scheduler_2], milestones=[warmup_steps])

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the starting epoch so files are named properly
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_loader)
            resume_step -= starting_epoch * len(train_loader)

    # Now we train the model
    for epoch in range(starting_epoch, args.epochs):
        model.train()
        total_train_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step
            active_dataloader = accelerator.skip_first_batches(train_loader, resume_step)
            overall_step += resume_step
        else:
            # After the first iteration though, we need to go back to the original dataloader
            active_dataloader = train_loader

        for image, label in active_dataloader:
            with accelerator.accumulate(model):
                if not args.no_aug:
                    image, label = mixup(image, label)
                with accelerator.autocast():
                    outputs = model(image)
                    loss = torch.nn.functional.cross_entropy(outputs, label)
                # We keep track of the loss at each epoch
                total_train_loss += loss.detach().float()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
                # IRE things
                if args.enable_ire:
                    optimizer.optimizer.set_descent(scheduler.get_last_lr()[0], args.lr, epoch >= args.ire_epochs)
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad()
                overall_step += 1
            # IRE things
            if args.enable_ire and overall_step % args.ire_interval == 1 and epoch >= args.ire_epochs:
                with accelerator.autocast():
                    outputs = model(image)
                    samp_dist = torch.distributions.Categorical(logits=outputs)
                    sampled_label = samp_dist.sample()
                    loss = torch.nn.functional.cross_entropy(outputs, sampled_label)
                accelerator.backward(loss)
                optimizer.optimizer.update_mask(epoch - args.ire_epochs, args.epochs - epoch)
                optimizer.zero_grad()

        model.eval()
        accurate = 0
        accurate_5 = 0
        num_elems = 0
        total_val_loss = 0
        for step, (image, label) in enumerate(val_loader):
            with torch.no_grad():
                outputs = model(image)
                loss = torch.nn.functional.cross_entropy(outputs, label)
                total_val_loss += loss.detach().float()
                _, predictions = outputs.topk(5, dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, label))
                accurate_preds = predictions[:, 0] == references
                accurate_5_preds = (predictions == references.view(-1, 1)).sum(dim=1)
                num_elems += accurate_preds.shape[0]
                accurate += accurate_preds.long().sum()
                accurate_5 += accurate_5_preds.long().sum()

        eval_metric = accurate.item() / num_elems
        eval_metric_5 = accurate_5.item() / num_elems
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"{datetime.datetime.now()} epoch {epoch}: top1 {100 * eval_metric:.2f} top5 {100 * eval_metric_5:.2f} ")
        accelerator.log(
            {
                "accuracy": 100 * eval_metric,
                "accuracy_5": 100 * eval_metric_5,
                "train_loss": total_train_loss.item() / len(train_loader),
                "val_loss": total_val_loss.item() / len(val_loader),
                "learning_rate": scheduler.get_last_lr()[0],
                "step": overall_step,
            },
            step=epoch,
        )

        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        if epoch % 10 == 9:
            accelerator.save_state(output_dir)

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Training script of ViT.")
    parser.add_argument(
        '-a',
        '--arch',
        metavar='ARCH',
        default='ViT-S/16',
        choices=model_zoo.model_dict.keys(),
        help='model architecture: ' + ' | '.join(model_zoo.model_dict.keys()) + ' (default: ViT-S/16)'
    )
    parser.add_argument(
        "--data_dir",
        default=".",
        help="The data folder on disk."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        '--project_name',
        type=str,
        default=None
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs` and relevent project information",
    )
    parser.add_argument(
        '--seed',
        default=None,
        type=int,
        help='seed for initializing training. '
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        default=64,
        type=int,
        metavar='N',
        help='batch size on a single GPU'
    )
    parser.add_argument(
        '--expected-batch-size',
        default=4096,
        type=int,
        help='expected total batch size'
    )
    parser.add_argument(
        '--optim',
        default='Lion',
        type=str,
        choices=['Lion', 'AdamW', 'SGD'],
        help='optim type'
    )
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=None,
        type=float,
        metavar='LR',
        help='initial learning rate for expected batch size',
        dest='lr'
    )
    parser.add_argument(
        '--weight-decay',
        default=None,
        type=float
    )
    parser.add_argument(
        '--momentum',
        default=None,
        type=float
    )
    parser.add_argument(
        '--grad-norm-clip',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--enable-ire',
        action='store_true',
    )
    parser.add_argument(
        '--ire-rank',
        default=0.,
        type=float
    )
    parser.add_argument(
        '--rank-increase',
        action='store_true',
    )
    parser.add_argument(
        '--prog',
        default=0.,
        type=float
    )
    parser.add_argument(
        '--prog-decay',
        action='store_true',
    )
    parser.add_argument(
        '--ire-interval',
        default=10,
        type=int
    )
    parser.add_argument(
        '--ire-epochs',
        default=5,
        type=int,
        metavar='N',
        help='number of total epochs to run'
    )
    parser.add_argument(
        '-j',
        '--workers',
        default=8, #np.clip(16, 2 * os.cpu_count() // 3, os.cpu_count()),
        type=int,
        metavar='N',
        help='number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--epochs',
        default=300,
        type=int,
        metavar='N',
        help='number of total epochs to run'
    )
    parser.add_argument(
        '--warmup-epochs',
        default=30,
        type=int,
        metavar='N',
        help='number of total epochs to run'
    )
    parser.add_argument(
        '--wandb-run-name',
        default=None,
        type=str
    )
    parser.add_argument(
        '--no-aug',
        action='store_true',
    )
    parser.add_argument(
        '--disable-log',
        action="store_true"
    )
    args = parser.parse_args()
    config = {}
    match (args.optim, args.no_aug):
        case ('AdamW', True):
            args.lr, args.weight_decay = args.lr or 1e-2, args.weight_decay or 0.1
        case ('Lion', True):
            args.lr, args.weight_decay = args.lr or 1e-3, args.weight_decay or 1.0
        case ('AdamW', False):
            args.lr, args.weight_decay = args.lr or 3e-3, args.weight_decay or 0.1
        case ('Lion', False):
            args.lr, args.weight_decay = args.lr or 3e-4, args.weight_decay or 1.0
    if args.wandb_run_name is None:
        args.wandb_run_name = f'{args.arch}_{args.lr}'
        if args.enable_ire:
            args.wandb_run_name += f'_ire_{args.ire_rank}_{args.prog}_{args.ire_epochs}'
    if args.output_dir is None:
        args.output_dir = args.wandb_run_name.replace('/', '_')
    training_function(config, args)


if __name__ == "__main__":
    main()
