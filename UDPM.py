import torch
import numpy as np
import utils
import lpips

class UDPM:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.shapes = torch.flip(((config.img_size / config.diffusion_scale ** config.diffusion_steps) *
                                  2 ** torch.arange(0, config.diffusion_steps + 1)).int().to(self.device), dims=(0,))
        self.w = utils.get_box(supp=(config.diffusion_scale,) * 2, size=(self.shapes[-2],) * 2).to(
            self.device)
        if config.w_norm_type == 'norm':
            self.w = self.w / self.w.norm()
        else:
            self.w = self.w / self.w.sum()

        if 't_start' in config and 't_end' in config and self.config.t_type == 'continuous':
            ts = torch.linspace(config.t_start, config.t_end, steps=self.config.diffusion_steps).to(self.device)
            self.delta_t = ts[1] - ts[0]
            print(f"ts = {ts.data}")
            self.ts = ts.flip(0)
            self.ts = torch.cat((self.ts, torch.zeros_like(ts[0:1])), dim=0)
            self.alphas_cumprod = self.alpha_bar(ts).to(self.device)
            self.alphas_cumprod = torch.cat((torch.ones(1, device=self.device), self.alphas_cumprod), dim=0)
            print(f"alpha_bar = {self.alphas_cumprod.data}")
            self.alphas = self.alphas_cumprod[1:] / self.alphas_cumprod[:-1]
            self.alphas = torch.cat((self.alphas_cumprod[0:1], self.alphas))
            self.betas = 1 - self.alphas
            print(f"alphas = {self.alphas.data}")
        elif self.config.t_type == 'discrete':
            ts = (torch.arange(1, self.config.diffusion_steps+1, device=self.device) - 0.5)/self.config.diffusion_steps
            self.delta_t = ts[1] - ts[0]
            print(f"ts = {ts.data} | delta t = {self.delta_t}")
            self.ts = ts.flip(0)
            self.ts = torch.cat((self.ts, torch.zeros_like(ts[0:1])), dim=0)
            self.alphas_cumprod = self.alpha_bar(ts).to(self.device)
            self.alphas_cumprod = torch.cat((torch.ones(1, device=self.device), self.alphas_cumprod), dim=0)
            print(f"alpha_bar = {self.alphas_cumprod.data}")
            self.alphas = self.alphas_cumprod[1:] / self.alphas_cumprod[:-1]
            self.alphas = torch.cat((self.alphas_cumprod[0:1], self.alphas))
            self.betas = 1 - self.alphas
            print(f"alphas = {self.alphas.data}")

        if 'lpips_weight' in config and config.lpips_weight > 0:
            self.percep_loss = lpips.LPIPS(net='vgg').to(config.device_gpu)
            self.resizer = torch.nn.Upsample(size=self.shapes[0], mode='bicubic')

    def alpha_bar(self, t):
        assert torch.all(0 <= t) and torch.all(t <= 1)
        if self.config.noise_scheduler == 'exp_gauss':
            return torch.exp(-t**2/(2*0.22**2)) * self.w.sum()**(-2 * self.get_scale_factor(t))
        elif self.config.noise_scheduler == 'magic':
            return torch.exp(-(t%(1/self.config.diffusion_steps))*5*self.config.diffusion_steps*np.log(10))
        elif self.config.noise_scheduler == 'magic_cosine':
            return torch.cos((0.5 * torch.pi * (t.double()%(1/self.config.diffusion_steps)) * self.config.diffusion_steps)).float()
        elif self.config.noise_scheduler == 'magic_gauss':
            return torch.exp(-(t%(1/self.config.diffusion_steps) * self.config.diffusion_steps)**2/(2*0.22**2))
        elif self.config.noise_scheduler == 'gauss':
            return torch.exp(-t**2/(2*0.22**2))
        if self.config.noise_scheduler == 'exp_gauss01':
            return torch.exp(-t**2/(2*0.1**2)) * self.w.sum()**(-2 * self.get_scale_factor(t))
        elif self.config.noise_scheduler == 'gauss015':
            return torch.exp(-t**2/(2*0.15**2))
        elif self.config.noise_scheduler == 'cosine':
            return torch.cos(0.5 * torch.pi * t.double())
        elif self.config.noise_scheduler == 'sqrt_cosine':
            return torch.sqrt(torch.cos(0.5 * torch.pi * t.double())).float()
        elif self.config.noise_scheduler == 'linear':
            return 1 - t
        elif self.config.noise_scheduler == 'cylinear':
            return ((1 - t) % (1/self.config.diffusion_steps)) / self.w.sum()**(2 * self.get_scale_factor(t))
        elif self.config.noise_scheduler == 'exp':
            return torch.exp(-0.5 * t)
        elif self.config.noise_scheduler == 'pow':
            return self.w.sum() ** (-10*t)
        elif self.config.noise_scheduler == 'constant':
            return torch.tensor(0.1, device=t.device) * self.w.sum()**(-2 * self.get_scale_factor(t))
        else:
            AssertionError("Unknown noise scheduler")

    def get_scale_factor(self, t):
        L = torch.tensor(self.config.diffusion_steps, device=t.device)
        return torch.ceil(t * L).long()

    def get_continuous_from_idx(self, t_idx):
        return t_idx.double() / self.config.diffusion_steps

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
        for k in range(t_idx[0]):
            x = self.apply_H(x)
        return x / self.w.sum()**t_idx if normalized else x

    def apply_H_conj_t(self, x, t, normalized=False):
        t_idx = self.get_scale_factor(t)
        for k in range(t_idx[0]):
            x = self.apply_H_conj(x)
        return x * self.w.sum()**t_idx if normalized else x

    def get_sigma_scalar(self, t):
        alpha_t = self.alpha_bar(t) / self.alpha_bar(t - self.delta_t)
        beta_t = 1 - alpha_t
        return 1 / (alpha_t / beta_t + 1 / (1 - self.alpha_bar(t - self.delta_t)))

    def apply_sigma_0p5(self, x, t):
        Sigma0p5 = torch.sqrt(self.get_sigma_scalar(t))
        return x * Sigma0p5

    def apply_sigma(self, x, t):
        Sigma = self.get_sigma_scalar(t)
        return x * Sigma

    def get_xt(self, x0, t):
        Htx0 = self.apply_Ht(x0, t, normalized=False)
        mean = torch.sqrt(self.alpha_bar(t)) * Htx0
        eps = torch.randn_like(mean)
        return mean + torch.sqrt(1 - self.alpha_bar(t)) * eps, eps

    def step_SR(self, model, x0, gScaler, optimizer, disc=None, cls=None):
        model.train()
        model = model.requires_grad_(True)
        if self.config.t_type == 'continuous':
            t = torch.rand(1, device=x0.device)
        elif self.config.t_type == 'discrete':
            t = (torch.randint(low=1, high=self.config.diffusion_steps + 1, size=(1, ), device=x0.device) - 0.5)/self.config.diffusion_steps
        # Prepare input:
        with torch.no_grad():
            xt, eps = self.get_xt(x0=x0, t=t.reshape(-1, 1, 1, 1))
            xt_up = self.apply_H_conj_t(xt, t)
            HT_1H_1_x0 = self.apply_H_conj_t(self.apply_Ht(x0, torch.clamp(t - 1 / self.config.diffusion_steps, 0, 1)),
                                        torch.clamp(t - 1 / self.config.diffusion_steps, 0, 1))
            if 'cat' in self.config.disc_type or 'res' in self.config.disc_type:
                HTH_x0 = self.apply_H_conj_t(self.apply_Ht(x0, t), t)

        # Train SR network:
        with torch.autocast(enabled=self.config.use_amp, device_type='cuda', dtype=torch.bfloat16 if self.config.use_amp else torch.float32):
            model_out = model(xt_up, self.config.time_scale*t, cls if self.config.classes_num > 1 else None)

            if self.config.obj_type == "L1":
                loss = torch.nn.functional.l1_loss(model_out, HT_1H_1_x0)
            elif self.config.obj_type == "L05":
                loss = torch.mean((torch.abs(model_out - HT_1H_1_x0) + 1e-3) ** 0.5)

            if self.config.lpips_weight > 0:
                loss_lpips = torch.mean(self.percep_loss(model_out, HT_1H_1_x0))
            else:
                loss_lpips = 0

            if self.config.disc_weight > 0 and disc is not None:
                disc = disc.requires_grad_(False)
                disc.eval()
                if 'cat' in self.config.disc_type:
                    fake_in = torch.cat((model_out, HTH_x0), dim=1)
                elif 'res' in self.config.disc_type:
                    fake_in = model_out - HTH_x0
                else:
                    fake_in = model_out
                fake = disc(fake_in, self.config.time_scale * t, cls if self.config.classes_num > 1 and self.config.class_disc else None)
                loss_g = torch.nn.functional.softplus(-fake).mean()
            else:
                loss_g = 0
        gScaler.scale(self.config.l1_weight * loss + self.config.disc_weight * loss_g + self.config.lpips_weight * loss_lpips).backward()
        gScaler.step(optimizer)
        gScaler.update()
        optimizer.zero_grad(set_to_none=True)

        return loss, loss_g


    def step_disc(self, model, x0, gScaler, optimizer, disc, cls=None):
        model.eval()
        model = model.requires_grad_(False)
        disc.train()
        disc = disc.requires_grad_(True)
        for p in disc.parameters():
            p.requires_grad = True
        if self.config.t_type == 'continuous':
            t = torch.rand(1, device=x0.device)
        elif self.config.t_type == 'discrete':
            t = (torch.randint(low=1, high=self.config.diffusion_steps + 1, size=(1, ), device=x0.device) - 0.5)/self.config.diffusion_steps
        # Prepare input:
        with torch.no_grad():
            xt, eps = self.get_xt(x0=x0, t=t.reshape(-1, 1, 1, 1))
            xt_up = self.apply_H_conj_t(xt, t)
            HT_1H_1_x0 = self.apply_H_conj_t(self.apply_Ht(x0, torch.clamp(t - 1 / self.config.diffusion_steps, 0, 1)),
                                        torch.clamp(t - 1 / self.config.diffusion_steps, 0, 1))
            if 'cat' in self.config.disc_type or 'res' in self.config.disc_type:
                HTH_x0 = self.apply_H_conj_t(self.apply_Ht(x0, t), t)
            with torch.autocast(enabled=self.config.use_amp, device_type='cuda', dtype=torch.bfloat16 if self.config.use_amp else torch.float32):
                model_out = model(xt_up, self.config.time_scale*t, cls if self.config.classes_num > 1 else None)

            if 'cat' in self.config.disc_type:
                fake_in = torch.cat((model_out.detach(), HTH_x0), dim=1)
                real_in = torch.cat((HT_1H_1_x0, HTH_x0), dim=1)
            elif 'res' in self.config.disc_type:
                fake_in = model_out.detach() - HTH_x0
                real_in = HT_1H_1_x0 - HTH_x0
            else:
                fake_in = model_out.detach()
                real_in = HT_1H_1_x0

        # Fake
        with torch.autocast(enabled=self.config.use_amp, device_type='cuda', dtype=torch.bfloat16 if self.config.use_amp else torch.float32):
            fake = disc(fake_in, self.config.time_scale * t, cls if self.config.classes_num > 1 and self.config.class_disc else None )
            loss_fake = torch.nn.functional.softplus(fake).mean()
        gScaler.scale(loss_fake / 2).backward()

        # Real
        with torch.autocast(enabled=self.config.use_amp, device_type='cuda', dtype=torch.bfloat16 if self.config.use_amp else torch.float32):
            real = disc(real_in, self.config.time_scale * t, cls if self.config.classes_num > 1 and self.config.class_disc else None )
            loss_real = torch.nn.functional.softplus(-real).mean()
        gScaler.scale(loss_real / 2).backward()

        gScaler.step(optimizer)
        gScaler.update()
        optimizer.zero_grad(set_to_none=True)

        return (loss_real + loss_fake)/2



    @torch.no_grad()
    def sample(self, model, batch_size=1, cls=None, eps=None):
        if self.config.save_inter:
            outs = []

        if eps is None:
            L = self.ts[0:1]
            xt = torch.sqrt(self.alpha_bar(L))*self.w.sum()**self.config.diffusion_steps*torch.ones(batch_size, 3,
                                                                                      self.shapes[-1], self.shapes[-1], dtype=torch.float32, device=self.device) + \
                    torch.sqrt(1 - self.alpha_bar(L))*torch.randn(batch_size, 3, self.shapes[-1], self.shapes[-1], dtype=torch.float32, device=self.device)
        else:
            xt = eps[0]
        for i, t in enumerate(self.ts):
            t = t.unsqueeze(0)
            xt_ = self.apply_H_conj_t(xt, t)
            with torch.autocast(enabled=False, device_type='cuda' if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                model_out = model(xt_, self.config.time_scale * t.repeat(xt_.shape[0]), cls)

            HTH_1_x0_hat = model_out

            Ht_1_x0 = self.apply_Ht(HTH_1_x0_hat, torch.clamp(t - self.delta_t, 0, 1), normalized=False) 

            if self.config.save_inter:
                outs.append(Ht_1_x0 / self.w.sum() ** self.get_scale_factor(torch.clamp(t - self.delta_t, 0, 1)))

            if t > self.delta_t:
                HT_xt = self.apply_H_conj(xt)
                alpha_t = self.alpha_bar(t)/self.alpha_bar(t - self.delta_t)
                beta_t = 1 - alpha_t
                Sigma_1_mu = torch.sqrt(alpha_t) / beta_t * HT_xt + \
                             torch.sqrt(self.alpha_bar(t - self.delta_t)) / (1 - self.alpha_bar(t - self.delta_t)) * Ht_1_x0
                mu = self.apply_sigma(Sigma_1_mu, t)
                e = self.apply_sigma_0p5(torch.randn_like(mu) if eps is None else eps[i+1], t)
                xt = mu + e
            else:
                xt = HTH_1_x0_hat
                break

        if self.config.save_inter:
            return outs
        else:
            return xt