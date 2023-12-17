## [UDPM] : Upsampling Diffusion Probabilistic Models

**This repository contains the official code and resources for training (comming soon) and sampling of [UDPM].**

### Getting Started

1. **Clone the repository:** `git clone https://github.com/shadyabh/UDPM.git`
2. **Install dependencies:** See `requirements.txt` for instructions.
3. **Download pre-trained models:** https://drive.google.com/drive/folders/1hYIpASc2zrkV2p-E9SKlAzDf0E6TzPIh?usp=sharing
4. **Training:** -- comming soon.
5. **Inference:** 

### Examples

* Generate images from CIFAR10:
  ```
  python sample.py --checkpoints_dir <<CHECKPOINTS DIRECTORY>> --batch_size 1 --suffix CIFAR10 --img_size 32 --classes_num 10 --use_ema --n_samples 10 --net_type NCSN --num_blocks_per_res 4 --model_channels 128 --model_ch_mult 2 2 2 --model_att_res 16 --t_type discrete --out_dir <<OUTPUT DIRECTORY>>
  ```
* Generate images from AFHQv2:
  ```
  python sample.py --checkpoints_dir <<CHECKPOINTS DIRECTORY>> --batch_size 1 --suffix AFHQv2 --img_size 64 --classes_num 0 --use_ema --n_samples 10 --net_type NCSN --num_blocks_per_res 4 --model_channels 128 --model_ch_mult 1 2 2 2 --model_att_res 16 --t_type discrete --out_dir <<OUTPUT DIRECTORY>>
  ```
