import torch
import torchvision
from PIL import Image
import argparse
import os
import UDPM
from model import UDPM_Net
import random
import numpy as np
import copy

args = argparse.ArgumentParser()
args.add_argument('--checkpoints_dir', type=str, default='')
args.add_argument('--from_itrs', type=int, default=None)
args.add_argument('--batch_size', type=int, default=1)
args.add_argument('--suffix', type=str, default='')
args.add_argument('--diffusion_steps', type=int, default=5)
args.add_argument('--diffusion_scale', type=int, default=2)
args.add_argument('--img_size', type=int, default=256)
args.add_argument('--seed1', type=int, default=0)
args.add_argument('--seed2', type=int, default=1)
args.add_argument('--seed3', type=int, default=2)
args.add_argument('--seed4', type=int, default=3)
args.add_argument('--classes_num', type=int, default=0)
args.add_argument('--use_ema', action='store_true')
args.add_argument('--save_inter', action='store_true')
args.add_argument('--save_jpg', action='store_true')
args.add_argument('--n_samples', type=int, default=8)
args.add_argument('--num_blocks_per_res', type=int, default=4)
args.add_argument('--out_dir', type=str, default='')
args.add_argument('--model_channels', type=int, default=128)
args.add_argument('--alphas', type=float, default=[], nargs='+')
args.add_argument('--sigmas', type=float, default=[], nargs='+')
args.add_argument('--model_ch_mult', type=int, nargs='+', default=[1, 4])
args.add_argument('--model_att_res', type=int, nargs='+', default=[32,16,8])
args.add_argument('--save_separate', action='store_true')
args.add_argument('--w_norm_type', type=str, default='norm', choices=['sum', 'norm'])
args.add_argument('--run_type', type=str, default='perturb', choices=['perturb', 'swap', 'interp'])
args.add_argument('--net_type', type=str, default='Dhariwal',
                  choices=['Dhariwal', 'DDPM', 'NCSN'])

args = args.parse_args()
args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

udpm = UDPM.UDPM(config=args, device=args.device)

# Load network:
model = UDPM_Net(in_shape=udpm.shapes[1], in_channels=3, num_blocks_per_res=args.num_blocks_per_res,
                 channel_mult=args.model_ch_mult, model_channels=args.model_channels,
                 attn_resolutions=args.model_att_res, out_channels=3, classes_num=args.classes_num,
                 class_dropout_prob=args.guidance_scale > 0, net_type=args.net_type, sf=args.diffusion_scale)
size = 0
for p in model.parameters():
    size += p.numel()

if args.from_itrs is None:
    directory = os.path.join(args.checkpoints_dir, f"model{'_ema' if args.use_ema else ''}_state_{args.suffix}.pt")
else:
    directories = [f for f in os.listdir(args.checkpoints_dir) if args.suffix in f and f"{args.from_itrs}" in f and ('.pt' in f or '.pth' in f)]
    if args.use_ema:
        directory = os.path.join(args.checkpoints_dir, [d for d in directories if '_ema' in d][0])
    else:
        directory = os.path.join(args.checkpoints_dir, [d for d in directories if '_ema' not in d][0])
    
state_dict = torch.load(directory, map_location="cpu")
print(f"Loaded checkpoint from {directory}")
model.load_state_dict(state_dict if args.use_ema else state_dict['model'])
model.eval()
model.to(args.device)
model = torch.nn.DataParallel(model)
print(f'Denoiser parameters: {size/1e6}M')

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)


if args.run_type == 'perturb':
    #### Perturbation ####
    torch.manual_seed(args.seed1)
    random.seed(args.seed1)
    np.random.seed(args.seed1)
    noises1 = [torch.randn(args.batch_size, 3, udpm.shapes[t], udpm.shapes[t], device=args.device) for t in range(args.diffusion_steps, 0, -1)]
    torch.manual_seed(args.seed2)
    random.seed(args.seed2)
    np.random.seed(args.seed2)
    noises2 = [torch.randn(args.batch_size, 3, udpm.shapes[t], udpm.shapes[t], device=args.device) for t in range(args.diffusion_steps, 0, -1)]
    gammas = [torch.linspace(1, a, 5) for a in [0.8, 0.1, 0]]
    x = []
    for t in range(args.diffusion_steps):
        for gamma in gammas[t]:
            noises = copy.deepcopy(noises1)
            noises[t] = gamma * noises1[t] + torch.sqrt(1 - gamma**2) * noises2[t]
            x_ = udpm.sample(model, eps=noises, cl=None)
            x.append(x_)
    torchvision.utils.save_image(0.5 * torch.cat(x, dim=0) + 0.5, os.path.join(args.out_dir, f"perturb_{args.seed1}_{args.seed2}{'.jpg' if args.save_jpg else '.png'}"),
                nrow=gammas[0].shape[0], pad_value=1.0)

elif args.run_type == 'perturb':
    #### Swapping ####
    torch.manual_seed(args.seed1)
    random.seed(args.seed1)
    np.random.seed(args.seed1)
    noises1 = [torch.randn(args.batch_size, 3, udpm.shapes[t], udpm.shapes[t], device=args.device) for t in range(args.diffusion_steps, 0, -1)]
    torch.manual_seed(args.seed2)
    random.seed(args.seed2)
    np.random.seed(args.seed2)
    noises2 = [torch.randn(args.batch_size, 3, udpm.shapes[t], udpm.shapes[t], device=args.device) for t in range(args.diffusion_steps, 0, -1)]
    gammas = torch.arange(0, args.diffusion_steps)

    x = x_ = udpm.sample(model, eps=noises1)
    for gamma in gammas:
        noises = noises2[:gamma] + noises1[gamma:gamma+1] + noises2[gamma+1:]
        x_ = udpm.sample(model, eps=noises)
        x = torch.cat((x, x_), dim=0)
    x_ = udpm.sample(model, eps=noises2[:gamma] + noises1[gamma:gamma+1] + noises2[gamma+1:])
    x = torch.cat((x, x_), dim=0)
    torchvision.utils.save_image(0.5 * x + 0.5, os.path.join(args.out_dir, f"generated_swap_{args.seed1}_{args.seed2}{'.jpg' if args.save_jpg else '.png'}"), nrow=args.diffusion_steps + 2, pad_value=1.0)


elif args.run_type == 'interp':
    #### Interpolation ####
    torch.manual_seed(args.seed1)
    random.seed(args.seed1)
    np.random.seed(args.seed1)
    noises1 = [torch.randn(1, 3, udpm.shapes[t], udpm.shapes[t], device=args.device) for t in range(args.diffusion_steps, -1, -1)]
    torch.manual_seed(args.seed2)
    random.seed(args.seed2)
    np.random.seed(args.seed2)
    noises2 = [torch.randn(1, 3, udpm.shapes[t], udpm.shapes[t], device=args.device) for t in range(args.diffusion_steps, -1, -1)]
    torch.manual_seed(args.seed3)
    random.seed(args.seed3)
    np.random.seed(args.seed3)
    noises3 = [torch.randn(1, 3, udpm.shapes[t], udpm.shapes[t], device=args.device) for t in range(args.diffusion_steps, -1, -1)]
    torch.manual_seed(args.seed4)
    random.seed(args.seed4)
    np.random.seed(args.seed4)
    noises4 = [torch.randn(1, 3, udpm.shapes[t], udpm.shapes[t], device=args.device) for t in range(args.diffusion_steps, -1, -1)]
    gammas_h = torch.linspace(0, 1, 6, device=args.device)
    gammas_v = torch.linspace(0, 1, gammas_h.shape[0], device=args.device)
    x = None
    for gamma_v in gammas_v:
        for gamma_h in gammas_h:
            noises = [gamma_v * (gamma_h * n1 + torch.sqrt(1-gamma_h**2) * n2) + torch.sqrt(1 - gamma_v**2) * (gamma_h * n3 + torch.sqrt(1-gamma_h**2) * n4)
                    for n1, n2, n3, n4 in zip(noises1, noises2, noises3, noises4)]
            x_ = udpm.sample(model, eps=noises)
            x = x_ if x is None else torch.cat((x, x_), dim=0)
    torchvision.utils.save_image(0.5 * x + 0.5, os.path.join(args.out_dir, 
            f"interp_4way_{args.seed1}_{args.seed2}_{args.seed3}_{args.seed4}{'.jpg' if args.save_jpg else '.png'}"), nrow=gammas_h.shape[0], normalize=False, pad_value=1)


print('\nFinished')