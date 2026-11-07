"""Классические итеративные методы (для сравнения) и градиентная доводка фазы."""
import math
import time

import torch

from .metrics import full_loss, ls_scale
from .physics import ROI, Fresnel, roi_crop


def _roi_mask(n_planes, n, rois, device):
    m = torch.zeros(n_planes, n, n, dtype=torch.bool, device=device)
    for k, (r, c) in enumerate(rois):
        m[k, r, c] = True
    return m


@torch.no_grad()
def gerchberg_saxton(tgt, prop: Fresnel, iters=200, rois=ROI, seed=0, callback=None):
    """Многоплоскостной алгоритм Герчберга-Сэкстона (вариант «global GS»):
    в каждой плоскости внутри информативной области модуль заменяется на целевой
    (с подобранным масштабом), вне её поле не трогаем; затем поля всех плоскостей
    возвращаются в плоскость ДОЭ, усредняются, и от суммы остаётся только фаза.
    tgt: (P, n, n) амплитуды в [0, 1]. Возвращает phi (n, n)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    P, n, _ = tgt.shape
    phi = (torch.rand(1, n, n, generator=g) * 2 - 1).to(tgt.device) * math.pi
    mask = _roi_mask(P, n, rois, tgt.device)
    t = tgt.unsqueeze(0)
    for it in range(iters):
        k = prop.forward_field(torch.polar(torch.ones_like(phi), phi))      # (1,P,n,n)
        amp = k.abs()
        s = ls_scale(roi_crop(amp, rois), roi_crop(t, rois))                 # (1,P,1,1)
        new_amp = torch.where(mask, t / s.clamp_min(1e-8), amp)
        k = torch.polar(new_amp, k.angle())
        h = prop.backward_field(k).sum(1)                                    # (1,n,n)
        phi = h.angle()
        if callback is not None:
            callback(it, phi[0])
    return phi[0]


def optimize_phase(tgt, prop: Fresnel, iters=300, lr=0.05, init=None, rois=ROI, seed=0,
                   callback=None, w_eff=0.0, cosine=False, decay_frac=0.3, warmup=None):
    """Прямая оптимизация фазы ДОЭ градиентным спуском (Adam) по той же функции
    потерь, что и у нейросети. init=None - случайная фаза (классический метод),
    init=phi_nn - доводка ответа нейросети. tgt: (P, n, n).
    warmup: шагов плавного разгона. По умолчанию 10 при готовом старте (первый шаг
    Adam сдвигает каждый пиксел почти на полный шаг и портит хороший ответ сети),
    0 - при случайном старте (классический метод остаётся как был)."""
    P, n, _ = tgt.shape
    if warmup is None:
        warmup = 10 if init is not None else 0
    if init is None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        init = (torch.rand(n, n, generator=g) * 2 - 1) * math.pi
    phi = init.detach().clone().to(tgt.device).unsqueeze(0).requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=lr)
    # cosine=True: первые (1 - decay_frac) шагов - постоянный шаг, затем он плавно
    # уменьшается до 2%: быстро в начале и точная сходимость в конце
    def _lr(i):
        w = min(1.0, (i + 1) / warmup) if warmup > 0 else 1.0
        if not cosine:
            return w
        start = (1 - decay_frac) * iters
        if i < start:
            return w
        return w * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (i - start) / max(1.0, iters - start))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr) if (cosine or warmup > 0) else None
    t = roi_crop(tgt.unsqueeze(0), rois)
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        loss, _ = full_loss(prop.forward_phase(phi), t, rois, w_eff)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if callback is not None:
            callback(it, phi.detach()[0])
    return phi.detach()[0]


def timed(fn, *a, **kw):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def polish_quantized(phi, tgt, prop: Fresnel, rounds=600, rois=ROI, w_eff=0.0, verbose=False):
    """Дискретная доводка уже 8-битного ДОЭ (то, что реально будет сдано в png).

    Квантование фазы на 256 уровней - заметный источник ошибки (ИСС 0.9999 -> 0.998).
    Алгоритм: фаза округляется до уровней png; затем по градиенту выбираются k
    пикселов, где сдвиг на ±1 уровень обещает наибольший выигрыш; шаг принимается,
    только если ошибка действительно уменьшилась (k растёт при успехе, уменьшается
    при неудаче). Перед этим подбирается общий сдвиг фазы (физически ничего не
    меняет, но меняет то, как пикселы округляются). Возвращает фазу ровно на уровнях png."""
    step = 2 * math.pi / 255
    t = roi_crop(tgt.unsqueeze(0), rois)

    def to_phi(u):
        return u.float() * step - math.pi

    def loss_of(u):
        with torch.no_grad():
            return full_loss(prop.forward_phase(to_phi(u)[None]), t, rois, w_eff)[0].item()

    def levels(p):                       # как phase_to_uint8, но на torch; 255 == 0 (это одна фаза)
        w = torch.remainder(p + math.pi, 2 * math.pi)
        return torch.round(w / step).long() % 255

    phi = phi.detach().to(tgt.device)
    best_u, best_l = None, float("inf")
    for c in torch.linspace(0, step, 9)[:-1].tolist():
        u = levels(phi + c)
        lo = loss_of(u)
        if lo < best_l:
            best_u, best_l = u, lo
    u, cur, start = best_u, best_l, best_l
    k, fails = 256, 0
    for r in range(rounds):
        ph = to_phi(u)[None].requires_grad_(True)
        lo, _ = full_loss(prop.forward_phase(ph), t, rois, w_eff)
        g, = torch.autograd.grad(lo, ph)
        g = g[0].ravel()
        idx = torch.topk(g.abs(), k).indices
        un = u.clone().ravel()
        un[idx] = (un[idx] - torch.sign(g[idx]).long()) % 255
        un = un.view_as(u)
        ln = loss_of(un)
        if ln < cur:
            u, cur, fails = un, ln, 0
            k = min(4096, int(k * 1.5) + 1)
        else:
            fails += 1
            k = max(1, k // 2)
            if k == 1 and fails > 12:
                break
    if verbose:
        print(f"[polish] ошибка после квантования {start:.6f} -> {cur:.6f} за {r + 1} раундов")
    return to_phi(u)
