import torch
import numpy as np
import utils
import os
import lpips
import torchvision

class UDPM:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.shapes = torch.flip(((config.img_size / config.diffusion_scale ** config.diffusion_steps) *
                                  2 ** torch.arange(0, config.diffusion_steps + 1)).int().to(self.device), dims=(0,))
        self.w = utils.get_box(supp=(config.diffusion_scale,) * 2, size=(config.diffusion_scale*2,) * 2).to(
            self.device)
        if config.w_norm_type == 'norm':
            self.w = self.w / self.w.norm()
        else:
            self.w = self.w / self.w.sum()
        ts = (torch.arange(1, self.config.diffusion_steps+1, device=self.device) - 0.5)/self.config.diffusion_steps
        self.delta_t = ts[1] - ts[0]
        print(f"ts = {ts.data} | delta t = {self.delta_t}")
        self.ts = ts.flip(0)
        self.ts = torch.cat((self.ts, torch.zeros_like(ts[0:1])), dim=0)
        self.alphas = torch.tensor(self.config.alphas, device=self.device)
        self.sigmas = torch.tensor(self.config.sigmas, device=self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        print(f"alpha_bar = {self.alphas_cumprod.data}")
        print(f"alphas = {self.alphas.data}")
        print(f"sigmas = {self.sigmas.data}")
        if 'lpips_weight' in config and len(config.lpips_weight) == config.diffusion_steps:
            self.percep_loss = utils.LPIPS(net='vgg').to(config.device_gpu)

    def get_scale_factor(self, t):
        L = torch.tensor(self.config.diffusion_steps, device=t.device)
        return torch.ceil(t * L).long()

    def apply_H(self, x):
        return utils.fft_Down_(x, self.w, self.config.diffusion_scale).real

    def apply_H_conj(self, x):
        W = utils.fft_circ(self.w, s=x.shape[-2:])
        W = utils.shift_by(W, 1 * (not self.config.diffusion_scale % 2))
        W_ = W.conj()
        w_ = torch.fft.ifftn(W_, dim=(-2, -1)).real
        return utils.fft_Up_(x, w_, self.config.diffusion_scale).real

    def apply_Ht(self, x, t, normalized=False):
        t_idx = self.get_scale_factor(t)
        outp = x.clone()
        for k in range(t_idx[0]):
            outp = self.apply_H(outp)
        return outp / self.w.sum()**t_idx if normalized else outp

    def apply_H_conj_t(self, x, t, normalized=False):
        t_idx = self.get_scale_factor(t)
        outp = x.clone()
        for k in range(t_idx[0]):
            outp = self.apply_H_conj(outp)
        return outp * self.w.sum()**t_idx if normalized else outp

    def get_sigma_scalar(self, t):
        t_idx = self.get_scale_factor(t)
        alpha_t = self.alphas[t_idx-1]
        sigma_t = self.sigmas[t_idx-1]
        sigma2_t_1_bar = self.alphas_cumprod[t_idx - 2] ** 2 * torch.sum(self.sigmas[:t_idx-1]**2/self.alphas_cumprod[:t_idx-1]**2)
        return 1 / (alpha_t ** 2 / sigma_t ** 2 + 1 / sigma2_t_1_bar)

    def apply_sigma_0p5(self, x, t):
        Sigma0p5 = torch.sqrt(self.get_sigma_scalar(t))
        return x * Sigma0p5
 
    def apply_sigma(self, x, t):
        Sigma = self.get_sigma_scalar(t)
        return x * Sigma

    def get_mu(self, xt, Ht_1_x0, t):
        HT_xt = self.apply_H_conj(xt)
        t_idx = self.get_scale_factor(t)
        alpha_t = self.alphas[t_idx-1]
        alpha_t_1_bar = self.alphas_cumprod[t_idx-2]
        sigma_t = self.sigmas[t_idx-1]
        sigma2_t_1_bar = self.alphas_cumprod[t_idx - 2] ** 2 * torch.sum(self.sigmas[:t_idx-1]**2/self.alphas_cumprod[:t_idx-1]**2)
        
        Sigma_1_mu = alpha_t / sigma_t ** 2 * HT_xt + alpha_t_1_bar / sigma2_t_1_bar * Ht_1_x0
        mu = self.apply_sigma(Sigma_1_mu, t)
        
        return mu

    def get_xt(self, x0, t, e=None):
        Htx0 = self.apply_Ht(x0, t, normalized=False)
        t_idx = self.get_scale_factor(t)
        mean = self.alphas_cumprod[t_idx-1] * Htx0
        if e is None:
            eps = torch.randn_like(mean)
        else:
            eps = e
        std = self.alphas_cumprod[t_idx - 1] ** 2 * torch.sum(self.sigmas[:t_idx]**2/self.alphas_cumprod[:t_idx]**2)
        return mean + torch.sqrt(std) * eps, eps
    
    def get_xt_(self, Htx0, t):
        '''The same as get_xt but without the downsampling'''
        t_idx = self.get_scale_factor(t)
        if torch.any(t_idx == 0):
            return Htx0, torch.zeros_like(Htx0)
        mean = self.alphas_cumprod[t_idx-1] * Htx0
        eps = torch.randn_like(mean)
        std = self.alphas_cumprod[t_idx - 1] ** 2 * torch.sum(self.sigmas[:t_idx]**2/self.alphas_cumprod[:t_idx]**2)
        return mean + torch.sqrt(std) * eps, eps
        

    def step_SR(self, model, x0, gScaler, optimizer, disc=None, cl=None):
        model.train()
        model = model.requires_grad_(True)
        t = (torch.randint(low=1, high=self.config.diffusion_steps + 1, size=(1, ), device=x0.device) - 0.5)/self.config.diffusion_steps
        # Prepare input:
        with torch.no_grad():
            xt, eps = self.get_xt(x0=x0, t=t.reshape(-1, 1, 1, 1))
            GT = self.apply_Ht(x0, torch.clamp(t - 1 / self.config.diffusion_steps, 0, 1))

        # Train SR network:
        with torch.autocast(enabled=self.config.use_amp, device_type='cuda', dtype=torch.bfloat16 if self.config.use_amp else torch.float32):
            model_out = model(xt, t, cl if self.config.classes_num > 1 else None)

            t_idx = self.get_scale_factor(t)

            if self.config.normalize_loss_inp:
                model_out = model_out / self.w.sum()**(t_idx-1)
                GT        = GT        / self.w.sum()**(t_idx-1)
                xt        = xt        / self.w.sum()**(t_idx-1)
                norm_factor = self.w.sum()**(t_idx-1)
            else:
                norm_factor = 1

            loss = torch.nn.functional.l1_loss(model_out, GT)          
            
            if self.config.lpips_weight[t_idx - 1] > 0:
                loss_lpips = torch.mean(self.percep_loss(model_out, GT, p=self.config.lpips_pow))
            else:
                loss_lpips = 0

            if len(self.config.disc_weight) > 0 and disc is not None:
                disc = disc.requires_grad_(False)
                disc.eval()

                fake_in = model_out
                
                if self.config.disc_loss_type == 'noisy':
                    fake_in += self.config.noisy_disc_std[t_idx - 1] * torch.randn_like(fake_in)
                elif self.config.disc_loss_type == 'diffusion':
                    fake_in = self.get_xt_(fake_in * norm_factor, torch.clamp(t - 1 / self.config.diffusion_steps, 0, 1))[0] / norm_factor
                
                if self.config.upsample_disc_inp:
                    fake_in = torch.nn.functional.interpolate(fake_in, size=None, scale_factor=int(self.config.diffusion_scale ** (t_idx-1)), mode='bilinear')

                fake = disc(fake_in, t, cl if self.config.classes_num > 1 and self.config.class_disc else None)
                
                loss_g = torch.nn.functional.softplus(-fake).mean()

            else:
                loss_g = 0
        gScaler.scale(self.config.l1_weight[t_idx - 1] * loss + self.config.disc_weight[t_idx-1] * loss_g + self.config.lpips_weight[t_idx - 1] * loss_lpips).backward()
        gScaler.step(optimizer)
        gScaler.update()
        optimizer.zero_grad(set_to_none=True)

        return loss, loss_g

    def step_disc(self, model, x0, gScaler, optimizer, disc, cl=None, t=None):
        model.eval()
        model = model.requires_grad_(False)
        disc.train()
        disc = disc.requires_grad_(True)
        for p in disc.parameters():
            p.requires_grad = True
        t = (torch.randint(low=1, high=self.config.diffusion_steps + 1, size=(1, ), device=x0.device) - 0.5)/self.config.diffusion_steps
        
        t_idx = self.get_scale_factor(t)

        # Prepare input:
        with torch.no_grad():
            xt, eps = self.get_xt(x0=x0, t=t.reshape(-1, 1, 1, 1))
            GT = self.apply_Ht(x0, torch.clamp(t - 1 / self.config.diffusion_steps, 0, 1))
                   
            with torch.autocast(enabled=self.config.use_amp, device_type='cuda', dtype=torch.bfloat16 if self.config.use_amp else torch.float32):
                model_out = model(xt, t, cl if self.config.classes_num > 1 else None)

            if self.config.normalize_loss_inp:
                model_out = model_out / self.w.sum()**(t_idx-1)
                GT        = GT        / self.w.sum()**(t_idx-1)
                xt        = xt        / self.w.sum()**(t_idx-1)
                norm_factor = self.w.sum()**(t_idx-1)
            else:
                norm_factor = 1
            
            fake_in = model_out.detach()
            real_in = GT.clone()

        # Fake
        if self.config.disc_loss_type == 'noisy':
            fake_in += self.config.noisy_disc_std[t_idx - 1] * torch.randn_like(fake_in)
        elif self.config.disc_loss_type == 'diffusion':
            fake_in = self.get_xt_(fake_in * norm_factor, torch.clamp(t - 1 / self.config.diffusion_steps, 0, 1))[0] / norm_factor

        if self.config.upsample_disc_inp:
            fake_in = torch.nn.functional.interpolate(fake_in, size=None, scale_factor=int(self.config.diffusion_scale ** (t_idx-1)), mode='bilinear')

        with torch.autocast(enabled=self.config.use_amp, device_type='cuda', dtype=torch.bfloat16 if self.config.use_amp else torch.float32):
            fake = disc(fake_in, t, cl if self.config.classes_num > 1 and self.config.class_disc else None, 
                        dp=self.config.disc_dropout[0] if len(self.config.disc_dropout) == 1 else self.config.disc_dropout[t_idx-1])
            loss_fake = torch.nn.functional.softplus(fake).mean()
        gScaler.scale(loss_fake / 2).backward()

        # Real
        if self.config.disc_loss_type == 'noisy':
            real_in += self.config.noisy_disc_std[t_idx - 1] * torch.randn_like(real_in)
        elif self.config.disc_loss_type == 'diffusion':
            real_in = self.get_xt_(real_in * norm_factor, torch.clamp(t - 1 / self.config.diffusion_steps, 0, 1))[0] / norm_factor

        if self.config.disc_grad_pen[t_idx-1] > 0:
            xt_1 = real_in
            xt_1.requires_grad = True
       
        if self.config.upsample_disc_inp:
            real_in = torch.nn.functional.interpolate(real_in, size=None, scale_factor=int(self.config.diffusion_scale ** (t_idx-1)), mode='bilinear')
    

        with torch.autocast(enabled=self.config.use_amp, device_type='cuda', dtype=torch.bfloat16 if self.config.use_amp else torch.float32):
            real = disc(real_in, t, cl if self.config.classes_num > 1 and self.config.class_disc else None,
                        dp=self.config.disc_dropout[0] if len(self.config.disc_dropout) == 1 else self.config.disc_dropout[t_idx-1])
            loss_real = torch.nn.functional.softplus(-real).mean()
        gScaler.scale(loss_real / 2).backward(retain_graph=self.config.disc_grad_pen[t_idx-1] > 0)
        
        if self.config.disc_grad_pen[t_idx-1] > 0:
            with torch.autocast(enabled=self.config.use_amp, device_type='cuda', dtype=torch.bfloat16 if self.config.use_amp else torch.float32):
                grad_real = torch.autograd.grad(outputs=real.sum(), inputs=xt_1, create_graph=True)[0]
                grad_real = grad_real.view(grad_real.size(0), -1)
                grad_penalty = (grad_real.norm(2, dim=1) ** 2).mean()
                grad_penalty = self.config.disc_grad_pen[t_idx-1] / 2 * grad_penalty
            gScaler.scale(grad_penalty).backward()

        gScaler.step(optimizer)
        gScaler.update()
        optimizer.zero_grad(set_to_none=True)

        return (loss_real + loss_fake)/2 



    @torch.no_grad()
    def sample(self, model, batch_size=1, cl=None, eps=None):
        if self.config.save_inter:
            outs = []

        x0 = torch.randn(batch_size, 3, self.shapes[0], self.shapes[0], dtype=torch.float32, device=self.device)
        L = self.ts[0:1]
        xt, __ = self.get_xt(x0, L, e=eps[0] if eps is not None else None)
        for i, t in enumerate(self.ts):
            t = t.unsqueeze(0)
            
            with torch.autocast(enabled=False, device_type='cuda', dtype=torch.bfloat16):
                model_out = model(xt, t.repeat(xt.shape[0]), cl)
            
            Ht_1_x0 = model_out
            if self.config.save_inter:
                outs.append(self.apply_H_conj_t(Ht_1_x0, t - 1/self.config.diffusion_steps))

            if t > self.delta_t:
                mu = self.get_mu(xt, Ht_1_x0, t)
                e = self.apply_sigma_0p5(torch.randn_like(mu) if eps is None else eps[i+1], t)
                xt = mu + e
            else:
                xt = model_out
                break

        if self.config.save_inter:
            return outs
        else:
            return xt