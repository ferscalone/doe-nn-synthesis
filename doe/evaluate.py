"""Ввод-вывод и итоговая оценка: ДОЭ.png -> 4 восстановленных png + метрики."""
import csv
import json
import os

import numpy as np
from PIL import Image

from .metrics import pcc_np, ssim_global_np, ssim_np, ssim_skimage
from .physics import (DISTANCES, ROI_INCLUSIVE, ROI_LITERAL, phase_to_uint8,
                      reconstruct_numpy, roi_crop_np, uint8_to_phase)


def load_targets(folder=None, files=None):
    """Читает 4 одноканальных png 128x128 (1.png..4.png). Возвращает uint8 (4,128,128).
    По заданию картинки 8-битные одноканальные; на всякий случай RGB/палитра
    переводятся в оттенки серого, а 16-битные - делением на 257 (65535 -> 255)."""
    if files is None:
        files = [os.path.join(folder, f"{i}.png") for i in range(1, 5)]
    imgs = []
    for f in files:
        im = Image.open(f)
        if im.mode != "L":
            print(f"[load] {os.path.basename(f)}: режим {im.mode}, ожидался L (8 бит, 1 канал) - конвертирую")
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            a = np.asarray(im).astype(np.float64)
            a = np.clip(np.round(a / 257.0 if a.max() > 255 else a), 0, 255).astype(np.uint8)
        else:
            a = np.asarray(im.convert("L"))
        assert a.shape == (128, 128), f"{f}: ожидается 128x128, получено {a.shape}"
        imgs.append(a)
    return np.stack(imgs)


def save_gray(path, arr_u8):
    Image.fromarray(np.asarray(arr_u8, dtype=np.uint8), mode="L").save(path)


def amplitude_to_uint8(amp, target_u8, roi, mode="fit"):
    """Перевод модуля поля в 8-битное изображение.
    fit: яркость подбирается методом наименьших квадратов по информативной
         области (как подбор экспозиции камеры); КК от этого не зависит.
    max: максимум модуля в информативной области -> 255 (без знания цели).
    frame: максимум модуля во всём кадре 128x128 -> 255 (как mat2gray; без знания цели).
    best: как fit, но «экспозиция» уточняется перебором (±20%) так, чтобы сумма
          ИСС (оконного и глобального со слайда) 8-битной картинки была
          максимальной (КК от этого не меняется)."""
    r, c = roi
    a = amp[r, c].astype(np.float64)
    if mode in ("fit", "best"):
        t = target_u8[r, c].astype(np.float64)
        s = (a * t).sum() / ((a * a).sum() + 1e-12)
        if mode == "best":
            cands = s * np.linspace(0.8, 1.2, 41)
            scores = []
            for x in cands:
                q = np.clip(np.round(a * x), 0, 255)
                scores.append(ssim_np(q, t) + ssim_global_np(q, t))
            s = cands[int(np.argmax(scores))]
    elif mode == "frame":
        s = 255.0 / (amp.max() + 1e-12)
    else:
        s = 255.0 / (a.max() + 1e-12)
    return np.clip(np.round(amp * s), 0, 255).astype(np.uint8)


def evaluate_doe(doe_u8, targets_u8, out_dir=None, norm="fit", fftshift=True, verbose=True):
    """Главная функция проверки. Восстановление выполняется из 8-битного ДОЭ
    (т.е. ровно из того, что будет выведено на ПВМС), по полной формуле Френеля."""
    phi = uint8_to_phase(doe_u8)
    amps = reconstruct_numpy(phi, DISTANCES, fftshift=fftshift)
    recs = np.stack([amplitude_to_uint8(amps[k], targets_u8[k], ROI_INCLUSIVE[k], norm)
                     for k in range(4)])
    rows = []
    for k in range(4):
        r_, c_ = ROI_INCLUSIVE[k]
        eff = float((amps[k][r_, c_] ** 2).sum() / (amps[k] ** 2).sum())
        row = {"image": k + 1, "z_mm": round(DISTANCES[k] * 1e3, 1), "efficiency": eff}
        for tag, rois in (("", ROI_INCLUSIVE), ("_literal61", ROI_LITERAL)):
            a = roi_crop_np(recs, rois)[k]
            t = roi_crop_np(targets_u8, rois)[k]
            row["pcc" + tag] = pcc_np(a, t)
            row["ssim" + tag] = ssim_np(a, t)                     # оконный, как skimage
            row["ssim_global" + tag] = ssim_global_np(a, t)       # формула со слайда 3
            row["ssim_matlab" + tag] = ssim_np(a, t, pad="replicate")
        sk = ssim_skimage(roi_crop_np(recs)[k], roi_crop_np(targets_u8)[k])
        if sk:
            row.update(sk)
        # те же метрики при двух других способах перевести |K| в 8 бит (независимо от norm):
        r_, c_ = ROI_INCLUSIVE[k]
        for tag, md in (("_fit", "fit"), ("_frame", "frame")):
            img = amplitude_to_uint8(amps[k], targets_u8[k], ROI_INCLUSIVE[k], md)
            row["pcc" + tag] = pcc_np(img[r_, c_], targets_u8[k][r_, c_])
            row["ssim" + tag] = ssim_np(img[r_, c_], targets_u8[k][r_, c_])
        rows.append(row)
    summary = {"pcc_mean": float(np.mean([r["pcc"] for r in rows])),
               "ssim_mean": float(np.mean([r["ssim"] for r in rows])),
               "ssim_global_mean": float(np.mean([r["ssim_global"] for r in rows])),
               "ssim_matlab_mean": float(np.mean([r["ssim_matlab"] for r in rows])),
               "ssim_fit_mean": float(np.mean([r["ssim_fit"] for r in rows])),
               "ssim_frame_mean": float(np.mean([r["ssim_frame"] for r in rows]))}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_gray(os.path.join(out_dir, "doe.png"), doe_u8)
        for k in range(4):
            save_gray(os.path.join(out_dir, f"rec_{k + 1}.png"), recs[k])
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"per_image": rows, "summary": summary, "norm": norm}, f, indent=2, ensure_ascii=False)
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        save_overview(os.path.join(out_dir, "overview.png"), targets_u8, doe_u8, recs, rows)
    if verbose:
        print_table(rows, summary)
    return rows, summary, recs


def print_table(rows, summary):
    print(f"{'изобр.':>6} {'z, мм':>6} {'КК':>8} {'ИСС':>8} {'ИСС-глоб':>9} {'ИСС-MATLAB':>10} {'эфф.':>6}")
    for r in rows:
        print(f"{r['image']:>6} {r['z_mm']:>6} {r['pcc']:8.4f} {r['ssim']:8.4f} {r['ssim_global']:9.4f} "
              f"{r['ssim_matlab']:10.4f} {r['efficiency']:6.1%}")
    print(f"{'сред.':>6} {'':>6} {summary['pcc_mean']:8.4f} {summary['ssim_mean']:8.4f} "
          f"{summary['ssim_global_mean']:9.4f} {summary['ssim_matlab_mean']:10.4f}")
    print("ИСС - оконный (11x11, как skimage); ИСС-глоб - по формуле со слайда (одно окно на всю область); "
          "ИСС-MATLAB - оконный с краями как в MATLAB ssim()")


def save_overview(path, targets, doe, recs, rows):
    """Сводная картинка: верхний ряд - исходные, нижний - восстановленные, справа ДОЭ."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(2, 5, figsize=(15, 6.2))
    for k in range(4):
        ax[0, k].imshow(targets[k], cmap="gray", vmin=0, vmax=255)
        ax[0, k].set_title(f"исходное {k + 1}")
        ax[1, k].imshow(recs[k], cmap="gray", vmin=0, vmax=255)
        ax[1, k].set_title(f"z={rows[k]['z_mm']} мм  КК={rows[k]['pcc']:.3f}  ИСС={rows[k]['ssim']:.3f}", fontsize=9)
    ax[0, 4].imshow(doe, cmap="twilight", vmin=0, vmax=255)
    ax[0, 4].set_title("ДОЭ (фаза, 0..255)")
    ax[1, 4].axis("off")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def phase_to_doe_u8(phi):
    return phase_to_uint8(np.asarray(phi))
