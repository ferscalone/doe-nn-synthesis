"""Метрики (КК Пирсона, ИСС/SSIM) и функция потерь."""
import numpy as np
import torch
import torch.nn.functional as F


def _gauss_window(size=11, sigma=1.5, device=None, dtype=torch.float32):
    g = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_torch(a, b, data_range=1.0, win_size=11, sigma=1.5, k1=0.01, k2=0.03, pad="valid"):
    """SSIM (Wang et al., 2004) с гауссовым окном 11x11, sigma=1.5.
    pad="valid": окна только целиком внутри картинки - как skimage(gaussian_weights=True,
                 use_sample_covariance=False) [основной вариант];
    pad="replicate": края дополняются повтором, среднее по всем пикселам - как ssim() в MATLAB.
    a, b: (..., H, W). Возвращает (...)."""
    shape = a.shape
    a = a.reshape(-1, 1, *shape[-2:])
    b = b.reshape(-1, 1, *shape[-2:])
    if pad == "replicate":
        r = win_size // 2
        a = F.pad(a, (r, r, r, r), mode="replicate")
        b = F.pad(b, (r, r, r, r), mode="replicate")
    w = _gauss_window(win_size, sigma, a.device, a.dtype)
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)
    saa = F.conv2d(a * a, w) - mu_a ** 2
    sbb = F.conv2d(b * b, w) - mu_b ** 2
    sab = F.conv2d(a * b, w) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return s.mean(dim=(-3, -2, -1)).reshape(shape[:-2])


def pcc_torch(a, b, eps=1e-8):
    """Коэффициент корреляции Пирсона по последним двум осям."""
    a = a - a.mean(dim=(-2, -1), keepdim=True)
    b = b - b.mean(dim=(-2, -1), keepdim=True)
    num = (a * b).sum(dim=(-2, -1))
    den = torch.sqrt((a * a).sum(dim=(-2, -1)) * (b * b).sum(dim=(-2, -1)) + eps)
    return num / den


def ls_scale(amp, tgt, eps=1e-8):
    """Масштаб s, минимизирующий ||s*amp - tgt||^2 (подбор «экспозиции»)."""
    return (amp * tgt).sum(dim=(-2, -1), keepdim=True) / ((amp * amp).sum(dim=(-2, -1), keepdim=True) + eps)


def doe_loss(amp, tgt, w_mse=1.0, w_pcc=1.0, w_ssim=1.0):
    """amp, tgt: (B, P, h, w) - модуль восстановленного поля и целевая амплитуда
    в информативных областях (tgt в [0, 1]). Масштаб amp подбирается
    аналитически, поэтому потеря зависит только от формы распределения."""
    a = amp * ls_scale(amp, tgt)
    mse = ((a - tgt) ** 2).mean()
    pcc = pcc_torch(a, tgt).mean()
    ssim = ssim_torch(a, tgt, data_range=1.0).mean()
    loss = w_mse * mse * 10.0 + w_pcc * (1 - pcc) + w_ssim * (1 - ssim)
    return loss, {"mse": mse.item(), "pcc": pcc.item(), "ssim": ssim.item()}


def efficiency(amp_full, amp_roi):
    """Дифракционная эффективность: доля энергии плоскости k, попавшая в её
    информативную область. amp_full (B,P,n,n), amp_roi (B,P,h,w) -> (B,P)."""
    return (amp_roi ** 2).sum(dim=(-2, -1)) / ((amp_full ** 2).sum(dim=(-2, -1)) + 1e-12)


def frame_norm_loss(amp_full, tgt_roi, rois):
    """Та же ошибка, но восстановление нормировано на максимум ВСЕГО кадра каждой
    плоскости (как сделал бы mat2gray / a / a.max()), без подбора яркости.
    Заставляет держать свет вне информативных областей не ярче нужного уровня -
    тогда метрики не зависят от того, как жюри нормирует восстановление."""
    from .physics import roi_crop
    a = roi_crop(amp_full / (amp_full.amax(dim=(-2, -1), keepdim=True) + 1e-8), rois)
    return 10 * ((a - tgt_roi) ** 2).mean() + (1 - pcc_torch(a, tgt_roi).mean()) \
        + (1 - ssim_torch(a, tgt_roi, data_range=1.0).mean())


def full_loss(amp_full, tgt_roi, rois, w_eff=0.0, w_frame=0.0):
    """doe_loss + необязательный штраф за низкую эффективность (w_eff > 0
    заставляет держать свет внутри областей ценой небольшой потери метрик)
    + необязательная ошибка при нормировке по всему кадру (w_frame > 0,
    используется в доводке для устойчивости к способу нормировки у жюри)."""
    from .physics import roi_crop
    amp_roi = roi_crop(amp_full, rois)
    loss, m = doe_loss(amp_roi, tgt_roi)
    if w_frame > 0:
        loss = loss + w_frame * frame_norm_loss(amp_full, tgt_roi, rois)
    if w_eff > 0:
        eff = efficiency(amp_full, amp_roi).mean()
        loss = loss + w_eff * (1 - eff)
        m["eff"] = eff.item()
    return loss, m


# ---- Метрики для итогового отчёта (по 8-битным картинкам) ---------------------
def pcc_np(a, b):
    a = np.array(a, np.float64).ravel()      # копия: не портим входной массив
    b = np.array(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def ssim_np(a, b, data_range=255.0, pad="valid"):
    """SSIM на гауссовом окне (та же реализация, что и при обучении)."""
    ta = torch.as_tensor(np.asarray(a, np.float64))
    tb = torch.as_tensor(np.asarray(b, np.float64))
    return float(ssim_torch(ta, tb, data_range=data_range, pad=pad))


def ssim_global_np(a, b, data_range=255.0, k1=0.01, k2=0.03):
    """ИСС ровно по формуле со слайда 3: средние, дисперсии и ковариация
    считаются по всей сравниваемой области сразу (одно «окно» на всю область)."""
    a = np.array(a, np.float64).ravel()
    b = np.array(b, np.float64).ravel()
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return float((2 * ma * mb + c1) * (2 * cov + c2) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))


def ssim_skimage(a, b, data_range=255.0):
    """Сверка с библиотекой scikit-image, если она установлена. Возвращает два
    варианта: с гауссовым окном (как MATLAB) и с настройками skimage по умолчанию."""
    try:
        from skimage.metrics import structural_similarity as ss
    except ImportError:
        return None
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    g = ss(a, b, data_range=data_range, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    d = ss(a, b, data_range=data_range)
    return {"ssim_skimage_gauss": float(g), "ssim_skimage_default": float(d)}
