import torch
import torchvision
from PIL import Image
import argparse
import os
import UDPM
from model import UDPM_Net
import random
import numpy as np

args = argparse.ArgumentParser()
args.add_argument('--checkpoints_dir', type=str, default='')
args.add_argument('--from_itrs', type=int, default=None)
args.add_argument('--batch_size', type=int, default=1)
args.add_argument('--suffix', type=str, default='')
args.add_argument('--diffusion_steps', type=int, default=5)
args.add_argument('--diffusion_scale', type=int, default=2)
args.add_argument('--img_size', type=int, default=256)
args.add_argument('--seed', type=int, default=0)
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
args.add_argument('--net_type', type=str, default='NCSN',
                  choices=['Dhariwal', 'DDPM', 'NCSN'])

args = args.parse_args()
args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

udpm = UDPM.UDPM(config=args, device=args.device)

# Load network:
model = UDPM_Net(in_shape=udpm.shapes[1], in_channels=3, num_blocks_per_res=args.num_blocks_per_res,
                 channel_mult=args.model_ch_mult, model_channels=args.model_channels,
                 attn_resolutions=args.model_att_res, out_channels=3, classes_num=args.classes_num, net_type=args.net_type, sf=args.diffusion_scale)
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

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.classes_num == 0:
    args.classes_num = 1

assert args.batch_size <= args.n_samples//args.classes_num

for c in range(args.classes_num):
    if args.save_separate and not os.path.isdir(os.path.join(args.out_dir, f"{c:04d}")):
            os.mkdir(os.path.join(args.out_dir, f"{c:04d}"))

    for i in range(0, args.n_samples//args.classes_num, args.batch_size):

        if args.classes_num > 1:
            cl = torch.randint(0, args.classes_num, (args.batch_size,)).to(args.device)
            cl = c * torch.ones_like(cl)
        else:
            cl = None

        if args.save_inter:
            outs = udpm.sample(model, batch_size=args.batch_size, cl=cl)
        else:
            x = udpm.sample(model, batch_size=args.batch_size, cl=cl)

        if cl is not None:
            name = f"{i}_cl{str(cl[0].item())}"
        else:
            name = str(i)
        if args.use_ema:
            name += '_ema'
        name += f"_{args.suffix}"
        if args.save_inter:
            [torchvision.utils.save_image(0.5 * outs[args.diffusion_steps - t - 1] + 0.5, args.out_dir + f'/model_out_{name}_{t}.png',
                                          nrow=4, normalize=False) for t in range(args.diffusion_steps-1, -1, -1)]
        else:
            if args.save_separate:
                for b, x_ in enumerate(x):
                    torchvision.utils.save_image(0.5 * x_ + 0.5, os.path.join(args.out_dir, f"{c:04d}", f"generated_{name}_{str(b)}{'.jpg' if args.save_jpg else '.png'}"), nrow=4,
                                             normalize=False)
            else:
                torchvision.utils.save_image(0.5 * x + 0.5, os.path.join(args.out_dir, f"generated_{name}{'.jpg' if args.save_jpg else '.png'}"), nrow=4,
                                         normalize=False)
        print(f"\rGenerated {(c) * args.n_samples//args.classes_num +  (i + 1)}/{args.n_samples}", end='')

print('\nFinished')
