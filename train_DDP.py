import torch
import torchvision.datasets.folder
from torch.cuda import amp
from torch.utils.data import DataLoader
import utils
from dataset import unlabeled_data, get_transforms
import time
import UDPM
from model import UDPM_Net
from discriminator import get_disc
import torch.distributed as dist
import copy
import random
import numpy as np
import argparse
import json
import os

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

args = argparse.ArgumentParser()
args.add_argument('--checkpoints_dir', type=str, default='')
args.add_argument('--data_dir', type=str, default='')
args.add_argument('--lr', type=float, default=1e-4)
args.add_argument('--lr_disc', type=float, default=1e-4)
args.add_argument('--EMA_weight', type=float, default=0.9999)
args.add_argument('--class_dropout_prob', type=float, default=0)
args.add_argument('--lpips_weight', type=float, default=[], nargs='+')
args.add_argument('--lpips_pow', type=float, default=2.0)
args.add_argument('--l1_weight', type=float, default=[], nargs='+')
args.add_argument('--alphas', type=float, default=[], nargs='+')
args.add_argument('--sigmas', type=float, default=[], nargs='+')
args.add_argument('--disc_weight', type=float, default=[], nargs='+')
args.add_argument('--batch_per_GPU', type=int, default=1)
args.add_argument('--num_workers', type=int, default=0)
args.add_argument('--epochs', type=int, default=1000)
args.add_argument('--save_every', type=int, default=5000)
args.add_argument('--suffix', type=str, default='')
args.add_argument('--diffusion_steps', type=int, default=5)
args.add_argument('--diffusion_scale', type=int, default=2)
args.add_argument('--dropout', type=float, default=0)
args.add_argument('--disc_dropout', type=float, nargs='+', default=[0,])
args.add_argument('--img_size', type=int, default=256)
args.add_argument('--disc_channels', type=int, default=64)
args.add_argument('--seed', type=int, default=0)
args.add_argument('--model_channels', type=int, default=128)
args.add_argument('--start_disc_after', type=int, default=0)
args.add_argument('--update_disc_every', type=int, default=1)
args.add_argument('--num_blocks_per_res', type=int, default=4)
args.add_argument('--channels_mult', type=int, nargs='+', default=[1, 4])
args.add_argument('--attention_res', type=int, nargs='+', default=[32,16,8])
args.add_argument('--local_rank', type=int, default=0)
args.add_argument('--training_steps', type=int, default=2000000)
args.add_argument('--print_interval', type=int, default=100)
args.add_argument('--use_amp', action='store_true')
args.add_argument('--class_disc', action='store_true')
args.add_argument('--upsample_disc_inp', action='store_true')
args.add_argument('--normalize_loss_inp', action='store_true')
args.add_argument('--classes_num', type=int, default=0)
args.add_argument('--warmup_steps', type=int, default=5000)
args.add_argument('--w_norm_type', type=str, default='norm', choices=['sum', 'norm'])
args.add_argument('--ddp_backend', type=str, default='nccl', choices=['nccl', 'mpi']) 
args.add_argument('--net_type', type=str, default='NCSN',
                  choices=['Dhariwal', 'DDPM', 'NCSN', 'NCSN_N3'])
args.add_argument('--disc_loss_type', type=str, default='standard',
                  choices=['standard', 'noisy', 'diffusion'])
args.add_argument('--disc_grad_pen',  type=float, nargs='+',default=[0.0,])
args.add_argument('--disc_grad_pen_factor', type=float, default=1)
args.add_argument('--noisy_disc_std', type=float, nargs='+',default=[0.0,])
args.add_argument('--noisy_disc_std_factor', type=float, default=1)
args = args.parse_args()


with open(os.path.join(args.checkpoints_dir, f"config_{args.suffix}.json"), 'wt') as f:
    json.dump(vars(args), f, indent=4)

args.world_size = torch.cuda.device_count()

if len(args.disc_weight) > 0:
    if len(args.disc_weight) == 1:
        args.disc_weight = args.disc_weight * args.diffusion_steps
    assert(len(args.disc_weight) == args.diffusion_steps)


if len(args.disc_grad_pen) == 1:
    args.disc_grad_pen = args.disc_grad_pen * args.diffusion_steps
assert(len(args.disc_grad_pen) == args.diffusion_steps)

if len(args.disc_weight) > 1:
    assert(len(args.disc_weight) == args.diffusion_steps)

if len(args.noisy_disc_std) > 0:
    if len(args.noisy_disc_std) == 1:
        args.noisy_disc_std = args.noisy_disc_std * args.diffusion_steps
    assert(len(args.noisy_disc_std) == args.diffusion_steps)

def train(args):
    if args.world_size > 1:
        rank = gpu = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend='nccl', world_size=args.world_size)
    else:
        gpu = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device_gpu = torch.device(gpu)
    else:
        device_gpu = torch.device("cpu")
    print(f"=> set cuda device = {gpu}")
    print(device_gpu)

    args.device_gpu = device_gpu

    if gpu == 0:
        print(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    udpm = UDPM.UDPM(config=args, device=device_gpu)

    if args.classes_num > 1:
        if "cifar" in args.data_dir.lower():
            dataset = torchvision.datasets.CIFAR10(args.data_dir, train=True, 
            transform=get_transforms(args.img_size), download=True)
        else:
            dataset = torchvision.datasets.folder.ImageFolder(args.data_dir, 
            transform=get_transforms(args.img_size))
    else:
        if "cifar" in args.data_dir.lower():
            dataset = torchvision.datasets.CIFAR10(args.data_dir, train=True, 
            transform=get_transforms(args.img_size))
        else:
            dataset = unlabeled_data(data_dir=args.data_dir, image_size=args.img_size)
    print(f"Found {len(dataset)} training images.")

    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=rank,
            drop_last=True,
            shuffle=True
        )
    else:
        train_sampler = None

    dataloader = DataLoader(dataset, batch_size=args.batch_per_GPU, sampler=train_sampler,
                            num_workers=args.num_workers, pin_memory=True, shuffle=None if args.world_size > 1 else True,
                            drop_last=True, persistent_workers=True if args.num_workers > 0 else False)


    model = UDPM_Net(in_shape=udpm.shapes[1], in_channels=3, num_blocks_per_res=args.num_blocks_per_res,
                     channel_mult=args.channels_mult, model_channels=args.model_channels, attn_resolutions=args.attention_res,
                     out_channels=3, classes_num=args.classes_num, net_type=args.net_type,
                     class_dropout_prob=args.class_dropout_prob, dropout=args.dropout, sf=args.diffusion_scale)
    
    num_pars = 0
    for p in model.parameters():
        num_pars += p.numel()
    print(f'Created model with {num_pars/1e6: .2f}M parameters')
    model = model.train().to(device_gpu)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.EMA_weight > 0 and gpu == 0:
        ema_state = copy.deepcopy(model.state_dict())

    if len(args.disc_weight) > 0:
        disc = get_disc(args)
        disc = disc.train().to(device_gpu)
        num_pars = 0
        for p in disc.parameters():
            num_pars += p.numel()
        print(f'Created discriminator with {num_pars/1e6: .2f}M parameters')
        optimizer_disc = torch.optim.Adam(disc.parameters(), lr=args.lr_disc)

    if not os.path.isdir(args.checkpoints_dir) and gpu == 0:
        os.mkdir(args.checkpoints_dir)


    gScaler = amp.GradScaler()

    start_epoch = 0
    training_steps = 0
    if os.path.isfile(os.path.join(args.checkpoints_dir, f"model_state"
                                                         f"_{args.suffix}.pt")):
        state_dict = torch.load(os.path.join(args.checkpoints_dir, f"model_"
                                                                   f"state_{args.suffix}.pt"),
                                map_location="cpu")
        model.load_state_dict(state_dict['model'])
        start_epoch = state_dict['epochs']
        training_steps = state_dict['optimizer']['state'][1]['step'] + 1
        if len(args.disc_weight) > 0:
            disc.load_state_dict(state_dict['discriminator'])
            optimizer_disc.load_state_dict(state_dict["opt_disc"])
            if optimizer_disc.param_groups[0]["lr"] != args.lr_disc and training_steps > args.warmup_steps:
                for g in optimizer_disc.param_groups:
                    g['lr'] = args.lr_disc
        optimizer.load_state_dict(state_dict['optimizer'])

        if 'gScaler' in state_dict.keys():
            gScaler.load_state_dict(state_dict['gScaler'])
        
        if optimizer.param_groups[0]["lr"] != args.lr and training_steps > args.warmup_steps:
            for g in optimizer.param_groups:
                g['lr'] = args.lr
        print(f"Loaded existing checkpoint model_state_"
              f"_{args.suffix}.pt, training steps = {training_steps}")
        if args.EMA_weight > 0 and gpu == 0:
            ema_state = torch.load(os.path.join(args.checkpoints_dir,
                                                     f"model_ema_state"
                                                     f"_{args.suffix}.pt"), map_location=device_gpu)

        args.noisy_disc_std = [s * args.noisy_disc_std_factor ** (training_steps // 1000) for s in args.noisy_disc_std]


    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],
                                                            find_unused_parameters=False)
        if len(args.disc_weight) > 0:
                disc = torch.nn.parallel.DistributedDataParallel(disc, device_ids=[gpu],
                                                            find_unused_parameters=False)
    else:
        model = torch.nn.DataParallel(model)
        if len(args.disc_weight) > 0:
            disc = torch.nn.DataParallel(disc)

    avg_loss = torch.tensor((0.0,), device=device_gpu)
    if len(args.disc_weight) > 0:
        avg_loss_d = torch.tensor((0.0,), device=device_gpu)
        avg_loss_g = torch.tensor((0.0,), device=device_gpu)
    optimizer.zero_grad(set_to_none=True)
    if len(args.disc_weight) > 0:
        optimizer_disc.zero_grad(set_to_none=True)
    tik = time.time()

    for epoch in range(start_epoch, args.epochs):
        for idx, data in enumerate(dataloader):
            with torch.no_grad():
                if training_steps <= args.warmup_steps:
                    for p in optimizer.param_groups:
                        p["lr"] = args.lr * training_steps / args.warmup_steps
                    if len(args.disc_weight) > 0:
                        for p in optimizer_disc.param_groups:
                            p["lr"] =  args.lr_disc * training_steps / args.warmup_steps

                if args.classes_num > 1:
                    x0, y = data[0].to(device_gpu, non_blocking=True), data[1].to(device_gpu, non_blocking=True)
                else:
                    x0 = data[0].to(device_gpu, non_blocking=True) if isinstance(data, list) else data.to(device_gpu, non_blocking=True)
                    y = None

                x0 = 2 * x0 - 1
            if len(args.disc_weight) > 0 and training_steps >= args.start_disc_after:
                if training_steps % args.update_disc_every == 0:
                    loss_d = udpm.step_disc(model=model, x0=x0, gScaler=gScaler, optimizer=optimizer_disc, disc=disc, cl=y)
                loss, loss_g = udpm.step_SR(model=model, x0=x0, gScaler=gScaler, optimizer=optimizer, disc=disc, cl=y)
            else:
                loss, __ = udpm.step_SR(model=model, x0=x0, gScaler=gScaler, optimizer=optimizer, disc=None, cl=y)
            training_steps += 1
            
            if training_steps % 1000 == 0:
                args.noisy_disc_std = [s * args.noisy_disc_std_factor for s in args.noisy_disc_std]
                args.disc_grad_pen = [args.disc_grad_pen_factor * pen for pen in args.disc_grad_pen]

            if gpu == 0:
                with torch.no_grad():
                    avg_loss += loss.detach()
                    if len(args.disc_weight) > 0 and training_steps > args.start_disc_after:
                        avg_loss_d += loss_d.detach()
                        avg_loss_g += loss_g.detach()
                if (training_steps+1) % args.print_interval == 0:
                    tok = time.time()
                    log_message = (f"GPU: {gpu}"
                        f" | Epoch = {epoch}/{args.epochs}"
                        f" | itrs = {training_steps} | "
                        f" | loss = {avg_loss.item()/args.print_interval: .5f} "
                        f" | time = {tok - tik: .3f} "
                        f" | lr = {optimizer.param_groups[0]['lr']:.3E}")
                     
                    if len(args.disc_weight) > 0:
                        log_message += f" | disc = {avg_loss_d.item()/args.print_interval}"
                        log_message += f" | gen = {avg_loss_g.item()/args.print_interval}"

                    print(log_message)
                    avg_loss = torch.tensor((0.0,), device=device_gpu)
                    avg_loss_d = torch.tensor((0.0,), device=device_gpu)
                    avg_loss_g = torch.tensor((0.0,), device=device_gpu)
                    tik = time.time()

            if args.EMA_weight > 0 and gpu == 0:
                with torch.no_grad():
                    ema_state = utils.EMA_update(ema_state, model.module.state_dict(), args.EMA_weight)

            if (training_steps + 1) % args.save_every == 0:
                if gpu == 0:
                    state_dict = {
                        'optimizer': optimizer.state_dict(),
                        'model': model.module.state_dict(),
                        'epochs': epoch,
                        'discriminator': disc.module.state_dict() if len(args.disc_weight) > 0 else None,
                        'opt_disc': optimizer_disc.state_dict() if len(args.disc_weight) > 0 else None,
                        'gScaler': gScaler.state_dict(),
                    }
                    torch.save(state_dict,
                               os.path.join(args.checkpoints_dir, f"model_"
                                                                  f"state_{args.suffix}.pt"))
                    if args.EMA_weight > 0:
                        torch.save(ema_state,
                               os.path.join(args.checkpoints_dir, f"model_"
                                                                  f"ema_state_{args.suffix}.pt"))

                if args.world_size > 1:
                    torch.distributed.barrier()



            if training_steps > args.training_steps:
                if args.world_size > 1:
                    dist.destroy_process_group()
                return
            if (training_steps + 1) % 10000 == 0:
                if gpu == 0:
                    state_dict = {
                        'optimizer': optimizer.state_dict(),
                        'model': model.module.state_dict(),
                        'epochs': epoch,
                        'discriminator': disc.module.state_dict() if len(args.disc_weight) > 0 else None,
                        'opt_disc': optimizer_disc.state_dict() if len(args.disc_weight) > 0 else None,
                        'gScaler': gScaler.state_dict(),
                    }
                    torch.save(state_dict, os.path.join(
                        args.checkpoints_dir,
                        f"model_state_{args.suffix}_"
                        f"epoch_{epoch}_itr_{state_dict['optimizer']['state'][1]['step']}.pt"))
                    if args.EMA_weight > 0:
                        torch.save(ema_state,
                               os.path.join(args.checkpoints_dir,
                                            f"model_ema_state_"
                                            f"{args.suffix}_epoch_{epoch}_itr_{state_dict['optimizer']['state'][1]['step']}.pt"))
                if args.world_size > 1:
                    torch.distributed.barrier()
    if args.world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    train(args)
