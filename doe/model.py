"""Нейросеть синтеза ДОЭ с физическим слоем внутри (physics-in-the-loop).

Схема (идея близка к DeepCGH, Eybposh et al., Opt. Express 2020):
  4 целевых изображения (4x128x128)
    -> U-Net A: предсказывает фазу поля psi_k в каждой из 4 плоскостей
    -> поля T_k*exp(j*psi_k) обратно распространяются к ДОЭ (точный оператор
       Френеля, не обучается) и складываются с обучаемыми весами
    -> phi0 = arg(сумма)  - первое приближение (как 1 шаг «умного» GS)
    -> U-Net B: по cos(phi0), sin(phi0), |сумма| и целям предсказывает поправку
    -> phi = phi0 + pi*tanh(delta)  - итоговая фаза ДОЭ (фазовый, |H| = 1)
Сеть обучается без готовых голограмм: потеря считается после прямого
распространения phi в 4 плоскости (самообучение через физическую модель).
"""
import torch
import torch.nn as nn

from .physics import Fresnel


def _gn(c):
    g = next(g for g in (8, 6, 4, 3, 2, 1) if c % g == 0)
    return nn.GroupNorm(g, c)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), _gn(cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), _gn(cout), nn.SiLU())
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class UNet(nn.Module):
    def __init__(self, cin, cout, base=32, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.inc = ConvBlock(cin, ch[0])
        self.down = nn.ModuleList([ConvBlock(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([ConvBlock(2 * ch[i], ch[i]) for i in reversed(range(depth))])
        self.pool = nn.AvgPool2d(2)
        self.out = nn.Conv2d(ch[0], cout, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(self.pool(skips[-1])))
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.out(x)


def _unit(z, eps=1e-6):
    """Единичный фазор z/|z| с гладким градиентом (без atan2)."""
    return z / torch.sqrt(z.real ** 2 + z.imag ** 2 + eps)


class DOENet(nn.Module):
    def __init__(self, base=32, base_ref=24, depth=4, n_planes=4, fftshift=True):
        super().__init__()
        self.prop = Fresnel(fftshift=fftshift)
        self.n_planes = n_planes
        self.unet_a = UNet(n_planes, 2 * n_planes, base, depth)
        self.unet_b = UNet(3 + n_planes, 1, base_ref, depth)
        self.plane_w = nn.Parameter(torch.ones(n_planes))
        nn.init.zeros_(self.unet_b.out.weight)      # старт: phi = phi0
        nn.init.zeros_(self.unet_b.out.bias)

    def forward(self, tgt):
        """tgt: (B, P, n, n) целевые амплитуды (>=0). Возвращает
        (phasor, phasor0): единичные фазоры итогового ДОЭ и первого приближения,
        оба (B, n, n), complex64. Фаза = torch.angle(phasor)."""
        B, P, n, _ = tgt.shape
        t = tgt / (tgt.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        o = self.unet_a(t).float().view(B, P, 2, n, n)
        e = _unit(torch.complex(o[:, :, 0], o[:, :, 1]))
        h = (self.prop.backward_field(t * e) * self.plane_w.view(1, P, 1, 1)).sum(1)
        h = h / (h.abs().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() + 1e-8)
        p0 = _unit(h)
        x = torch.cat([p0.real.unsqueeze(1), p0.imag.unsqueeze(1),
                       h.abs().unsqueeze(1).clamp(max=4.0), t], 1)
        d = torch.pi * torch.tanh(self.unet_b(x).float()[:, 0])
        p = p0 * torch.polar(torch.ones_like(d), d)
        return p, p0

    def forward_all(self, tgt):
        """Все промежуточные ответы (для глубокого контроля): [phasor0, phasor]."""
        p, p0 = self.forward(tgt)
        return [p0, p]

    def forward_phases(self, tgt):
        """То же, но фазами (вещественные тензоры) - удобно для нескольких GPU."""
        return [_safe_angle(p) for p in self.forward_all(tgt)]

    @torch.no_grad()
    def predict_phase(self, tgt):
        p, _ = self.forward(tgt)
        return torch.angle(p)

    def amplitudes(self, phasor):
        """Модули полей в 4 плоскостях для фазора ДОЭ (B, n, n)."""
        return self.prop.forward_field(phasor).abs()


# =============================================================================
# v2: развёрнутая (unrolled) сеть - «выученный итеративный алгоритм»
# =============================================================================
def _safe_angle(z, eps=1e-12):
    """arg(z) без бесконечного градиента в нуле."""
    return torch.atan2(z.imag, z.real + eps * (z.abs() < eps))


class UnrolledDOENet(nn.Module):
    """Идея (deep unrolling / «обучаемый Герчберг-Сэкстон»):

      phi_0 = начальное приближение (как в v1: U-Net A -> обратный Френель)
      для t = 1..T (каждый блок со своими весами):
          K_k   = Френель(phi)                    4 плоскости, точная физика
          GS-шаг: в информативной области модуль -> цель, обратно к ДОЭ -> h_gs
          невязка: (|K|-цель) с фазой K, обратно к ДОЭ -> g  (≈ градиент ошибки)
          CNN_t(cos/sin шага GS, |h_gs|, невязка g, cos/sin phi) -> delta
          phi <- phi + alpha_t * (шаг GS) + pi*tanh(delta)

    Без обучения (delta=0, alpha=1) блок - ровно одна итерация GS, т.е. сеть
    заведомо не хуже T итераций классики; обучение учит делать шаги намного
    «умнее». Всё дифференцируемо, сеть учится end-to-end через физику.
    """

    N_FEAT = 7

    def __init__(self, base=32, stage_base=32, stages=12, depth=3, depth_a=4, n_planes=4, fftshift=True,
                 grad_ckpt=True):
        super().__init__()
        from .physics import N, ROI
        self.prop = Fresnel(fftshift=fftshift)
        self.n_planes = n_planes
        self.grad_ckpt = grad_ckpt
        self.unet_a = UNet(n_planes, 2 * n_planes, base, depth_a)
        self.plane_w = nn.Parameter(torch.ones(n_planes))
        self.blocks = nn.ModuleList([UNet(self.N_FEAT, 1, stage_base, depth) for _ in range(stages)])
        for b in self.blocks:                      # старт: каждый блок = шаг GS
            nn.init.zeros_(b.out.weight)
            nn.init.zeros_(b.out.bias)
        self.alpha = nn.Parameter(torch.ones(stages))
        m = torch.zeros(n_planes, N, N, dtype=torch.bool)
        for k, (r, c) in enumerate(ROI):
            m[k, r, c] = True
        self.register_buffer("mask", m)
        self.rois = ROI

    def _init_phase(self, t):
        B, P, n, _ = t.shape
        o = self.unet_a(t).float().view(B, P, 2, n, n)
        e = _unit(torch.complex(o[:, :, 0], o[:, :, 1]))
        h = (self.prop.backward_field(t * e) * self.plane_w.view(1, P, 1, 1)).sum(1)
        return _safe_angle(h)

    def _physics(self, phi, t):
        from .metrics import ls_scale
        from .physics import roi_crop
        P = self.n_planes
        p = torch.polar(torch.ones_like(phi), phi)
        K = self.prop.forward_field(p)
        amp = K.abs()
        u = K / (amp + 1e-8)
        s = ls_scale(roi_crop(amp, self.rois), roi_crop(t, self.rois)).clamp_min(1e-8)
        tgt_amp = t / s                                          # цель в масштабе поля
        k_gs = torch.where(self.mask, tgt_amp * u, K)
        k_res = torch.where(self.mask, (amp - tgt_amp) * u, torch.zeros_like(K))
        B = phi.shape[0]
        back = self.prop.backward_field(torch.cat([k_gs, k_res], 0)).sum(1)   # один вызов на оба поля
        h_gs, g = back[:B], back[B:]
        rms = h_gs.abs().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() + 1e-8
        rel = p.conj() * h_gs                                    # шаг GS относительно phi
        d_gs = _safe_angle(rel)
        rg = p.conj() * g / rms
        feat = torch.stack([torch.cos(d_gs), torch.sin(d_gs), (h_gs.abs() / rms).clamp(max=4.0),
                            (rg.real * 4).clamp(-30, 30), (rg.imag * 4).clamp(-30, 30),
                            torch.cos(phi), torch.sin(phi)], 1)       # (ограничение - защита от выбросов в fp16)
        return feat, d_gs

    def _block(self, i, phi, t):
        feat, d_gs = self._physics(phi, t)
        d = self.blocks[i](feat).float()[:, 0]
        return phi + self.alpha[i] * d_gs + torch.pi * torch.tanh(d)

    def forward_phases(self, tgt):
        t = tgt / (tgt.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        phis = [self._init_phase(t)]
        for i in range(len(self.blocks)):
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                phi = checkpoint(self._block, i, phis[-1], t, use_reentrant=False)
            else:
                phi = self._block(i, phis[-1], t)
            phis.append(phi)
        return phis

    def forward_all(self, tgt):
        return [torch.polar(torch.ones_like(f), f) for f in self.forward_phases(tgt)]

    def forward(self, tgt):
        ps = self.forward_all(tgt)
        return ps[-1], ps[0]

    @torch.no_grad()
    def predict_phase(self, tgt):
        return self.forward_phases(tgt)[-1]

    def amplitudes(self, phasor):
        return self.prop.forward_field(phasor).abs()


def load_weights(model, state):
    """Загрузка весов; старые чекпойнты содержали константу prop.q - пропускаем её."""
    state = {k: v for k, v in state.items() if not k.endswith("prop.q")}
    model.load_state_dict(state)
    return model


def build_model(cfg):
    """Создаёт сеть по конфигу из чекпойнта (старые чекпойнты = v1)."""
    cfg = dict(cfg or {})
    arch = cfg.pop("arch", "v1")
    if arch == "v2":
        return UnrolledDOENet(**cfg)
    return DOENet(**cfg)
