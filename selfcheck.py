"""Самопроверка проекта (~1 мин на CPU):  python selfcheck.py

Проверяет физическую модель, кодирование фазы в png, метрики (против numpy и
scikit-image, если он установлен), модели и формат выходных файлов.
"""
import math
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

from doe.classic import gerchberg_saxton, optimize_phase, polish_quantized
from doe.evaluate import evaluate_doe, load_targets
from doe.metrics import pcc_np, ssim_global_np, ssim_np
from doe.model import DOENet, UnrolledDOENet, build_model, load_weights
from doe.physics import (DISTANCES, ROI, Fresnel, chirp_nyquist_ratio, output_pixel_pitch, phase_to_uint8,
                         reconstruct_numpy, uint8_to_phase)

HERE = os.path.dirname(os.path.abspath(__file__))
ok_all = True


def check(name, cond, info=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'OK' if cond else 'ОШИБКА'}] {name}" + (f"  ({info})" if info else ""))


torch.manual_seed(0)
print("Параметры задачи:")
for z in DISTANCES:
    print(f"  z = {z * 1e3:.1f} мм: шаг в плоскости изображения {output_pixel_pitch(z) * 1e6:.2f} мкм, "
          f"λz/(NΔx²) = {chirp_nyquist_ratio(z):.3f}")
check("квадратичная фаза дискретизована без наложения (λz/(NΔx²) ≥ 1)", all(chirp_nyquist_ratio(z) >= 1 for z in DISTANCES))

# --- физика --------------------------------------------------------------------
prop = Fresnel()
phi = (torch.rand(128, 128) * 2 - 1) * math.pi
a_t = prop.forward_phase(phi[None])[0].double().numpy()
a_n = reconstruct_numpy(phi.double().numpy())
c = (a_t * a_n).sum((1, 2)) / (a_n * a_n).sum((1, 2))
err = np.abs(a_n * c[:, None, None] - a_t).max() / a_t.max()
check("быстрая (torch) и полная (numpy, формула со слайда) модели совпадают", err < 1e-5, f"отн. ошибка {err:.1e}")
h = torch.polar(torch.ones_like(phi), phi)[None]
inv = (prop.backward_field(prop.forward_field(h)) - h[:, None]).abs().max().item()
check("обратное распространение - точная обратная операция", inv < 1e-5, f"{inv:.1e}")
check("центр кадра (нулевой порядок) в пикселе (64, 64), в зазоре между областями",
      np.unravel_index(np.argmax(np.abs(np.fft.fftshift(np.fft.fft2(np.ones((128, 128)))))), (128, 128)) == (64, 64))

# --- кодирование фазы ------------------------------------------------------------
u = phase_to_uint8(np.array([-math.pi, 0.0, math.pi - 1e-9]))
check("кодирование: -π -> 0, 0 -> 128, +π -> 255", [int(x) for x in u] == [0, 128, 255], str([int(x) for x in u]))
p = np.random.default_rng(0).uniform(-10, 10, (128, 128))
d = np.angle(np.exp(1j * (uint8_to_phase(phase_to_uint8(p)) - p)))
check("ошибка квантования фазы ≤ половины уровня (π/255)", np.abs(d).max() <= math.pi / 255 + 1e-9, f"{np.abs(d).max():.4f}")

# --- метрики -------------------------------------------------------------------
T = load_targets(os.path.join(HERE, "examples", "kit01_photos"))
A = T[3][ROI[3]].astype(float)
B = np.clip(A + np.random.default_rng(1).normal(0, 20, A.shape), 0, 255).round()
check("КК совпадает с numpy.corrcoef", abs(pcc_np(A, B) - np.corrcoef(A.ravel(), B.ravel())[0, 1]) < 1e-12)
A0 = A.copy(); pcc_np(A, B)
check("функция КК не портит входной массив", np.array_equal(A, A0))
check("ИСС(A, A) = 1 (оконный и глобальный)", abs(ssim_np(A, A) - 1) < 1e-12 and abs(ssim_global_np(A, A) - 1) < 1e-12)
try:
    from skimage.metrics import structural_similarity as ss
    ref = ss(A, B, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    check("оконный ИСС совпадает с scikit-image", abs(ssim_np(A, B) - ref) < 1e-9, f"{ssim_np(A, B):.6f} vs {ref:.6f}")
except ImportError:
    print("[--] scikit-image не установлен - сверка ИСС пропущена")

# --- входные картинки -----------------------------------------------------------
check("пример комплекта: 4 шт., 128x128, uint8", T.shape == (4, 128, 128) and T.dtype == np.uint8)
outside = [T[k].copy() for k in range(4)]
for k in range(4):
    outside[k][ROI[k]] = 0
check("вне информативных областей картинки чёрные", all(o.max() == 0 for o in outside))

# --- модели --------------------------------------------------------------------
t = torch.tensor(T / 255.0, dtype=torch.float32)
m2 = UnrolledDOENet(base=8, stage_base=8, stages=2).eval()
with torch.no_grad():
    phis = m2.forward_phases(t[None])
    # необученный блок (поправка 0, alpha 1) должен быть ровно шагом GS
    K = prop.forward_field(torch.polar(torch.ones_like(phis[0]), phis[0]))
    from doe.metrics import ls_scale
    from doe.physics import roi_crop
    amp = K.abs()
    s = ls_scale(roi_crop(amp), roi_crop(t[None])).clamp_min(1e-8)
    kg = torch.where(m2.mask, (t[None] / s) * K / (amp + 1e-8), K)
    gs1 = prop.backward_field(kg).sum(1).angle()
dd = torch.angle(torch.exp(1j * (phis[1] - gs1))).abs().max().item()
check("необученный блок v2 = ровно одна итерация GS", dd < 1e-4, f"{dd:.1e}")
m2.train()
outs = m2.forward_all(t[None])
from doe.metrics import full_loss  # noqa: E402
lo, _ = full_loss(m2.amplitudes(outs[-1]), roi_crop(t[None]), ROI)
lo.backward()
check("градиенты v2 конечны", all(torch.isfinite(q.grad).all() for q in m2.parameters() if q.grad is not None))

m1 = DOENet(base=8, base_ref=8)
sd = m1.state_dict()
sd["prop.q"] = torch.zeros(4, 128, 128, dtype=torch.complex64)        # как в старых чекпойнтах
try:
    load_weights(build_model({"base": 8, "base_ref": 8}), sd)
    check("старые чекпойнты v1 (с prop.q) читаются", True)
except Exception as ex:  # noqa: BLE001
    check("старые чекпойнты v1 (с prop.q) читаются", False, str(ex))

# --- классика и итоговые файлы ---------------------------------------------------
phi_gs = gerchberg_saxton(t, prop, iters=30)
phi_ad = optimize_phase(t, prop, iters=60)
phi_po = polish_quantized(phi_ad, t, prop, rounds=40)
lv = (phi_po + math.pi) / (2 * math.pi / 255)
check("после дискретной доводки фаза ровно на уровнях png", (lv - lv.round()).abs().max().item() < 1e-3)
with tempfile.TemporaryDirectory() as td:
    rows, summ, _ = evaluate_doe(phase_to_uint8(phi_po.numpy()), T, out_dir=td, verbose=False)
    files = ["doe.png", "rec_1.png", "rec_2.png", "rec_3.png", "rec_4.png", "metrics.csv", "metrics.json"]
    check("все файлы результата созданы", all(os.path.exists(os.path.join(td, f)) for f in files))
    fmt = all(Image.open(os.path.join(td, f)).mode == "L" and Image.open(os.path.join(td, f)).size == (128, 128)
              for f in files[:5])
    check("png: 128x128, 8 бит, один канал (режим L)", fmt)
    back = np.asarray(Image.open(os.path.join(td, "doe.png")))
    check("doe.png читается обратно без потерь", np.array_equal(back, phase_to_uint8(phi_po.numpy())))
print(f"GS 30 итер.: ИСС {evaluate_doe(phase_to_uint8(phi_gs.numpy()), T, verbose=False)[1]['ssim_mean']:.3f}; "
      f"Adam 60 + доводка: ИСС {summ['ssim_mean']:.3f}")
print("\nИТОГ:", "всё в порядке" if ok_all else "ЕСТЬ ОШИБКИ")
sys.exit(0 if ok_all else 1)
