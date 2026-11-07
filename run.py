"""Главная точка входа: синтез фазового ДОЭ для каждого комплекта из 4 изображений.

    python run.py --input ПАПКА_С_КОМПЛЕКТАМИ --output ПАПКА_РЕЗУЛЬТАТОВ [--mode best]

Вход (--input), любой из вариантов:
  * папка с подпапками-комплектами:  input/<комплект>/{4 файла .png}
  * одна папка ровно с 4 файлами .png (один комплект)
Файлы комплекта сортируются по имени «по-человечески» (1.png, 2.png, ..., 10.png),
изображение k соответствует расстоянию k: 24,1 / 25,3 / 26,5 / 27,7 мм.
Изображения: 128x128, 8 бит, оттенки серого; информативная часть картинки k - в углу k.

Выход (--output), для каждого комплекта папка <комплект>/:
  doe.png                  фазовый ДОЭ: PNG 128x128, 8 бит, grayscale; 0 = -pi, 255 = +pi
  rec_1.png .. rec_4.png   восстановленные изображения (модуль поля, Френель с одним БПФ),
                           нормированы на максимум всего кадра -> 255 (стандартно, без знания цели)
  metrics.csv / .json      SSIM и Pearson по информативным областям
и общие файлы summary.csv (по комплектам + среднее) и timing.json.

Режимы (--mode):
  fast      только нейросеть (один проход)
  balanced  нейросеть + 800 шагов устойчивой доводки + дискретная доводка 8-битного ДОЭ
  best      нейросеть + 2000 шагов устойчивой доводки + дискретная доводка 8-битного ДОЭ  [по умолчанию]
  max       нейросеть + 4000 шагов устойчивой доводки + дискретная доводка (ещё точнее, вдвое дольше)
  best_roi  прежний вариант: 1000 шагов без устойчивости - максимум SSIM, если яркость
            восстановления подбирается по информативной области, но плохо при нормировке по кадру

«Устойчивая» доводка оптимизирует ДОЭ сразу под два способа перевести модуль поля в 8 бит:
с подбором яркости по информативной области и с нормировкой на максимум всего кадра
(как mat2gray). Тогда результат не зависит от того, как именно нормирует проверяющий.
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import numpy as np
import torch

from doe.evaluate import evaluate_doe, load_targets
from doe.physics import ROI, phase_to_uint8
from synthesize import load_model, synthesize

W_FRAME = 0.5   # вес ошибки «нормировка по всему кадру» (подобран на dataset/samples, см. docs/tuning.md)
MODES = {  # режим -> (метод synthesize, шагов доводки, запасных стартов, w_frame, нормировка png)
    "fast": ("nn", 0, 0, 0.0, "frame"),
    "balanced": ("best", 800, 0, W_FRAME, "frame"),
    "best": ("best", 2000, 0, W_FRAME, "frame"),
    "max": ("best", 4000, 0, W_FRAME, "frame"),
    "best_roi": ("best", 1000, 0, 0.0, "best"),
}
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "doe_unrolled_v2.pt")


def natural_key(s):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def pngs(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")], key=natural_key)


def find_kits(root):
    """Список (имя комплекта, [4 пути к png])."""
    own = pngs(root)
    if len(own) >= 4:
        if len(own) > 4:
            print(f"[run] в {root} {len(own)} png - беру первые 4 по имени: {own[:4]}")
        return [(os.path.basename(os.path.normpath(root)) or "kit", [os.path.join(root, f) for f in own[:4]])]
    kits = []
    for d in sorted(os.listdir(root), key=natural_key):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            fs = pngs(p)
            if len(fs) >= 4:
                if len(fs) > 4:
                    print(f"[run] {d}: {len(fs)} png - беру первые 4 по имени: {fs[:4]}")
                kits.append((d, [os.path.join(p, f) for f in fs[:4]]))
            elif fs:
                print(f"[run] {d}: только {len(fs)} png (нужно 4) - пропускаю")
    return kits


def check_layout(name, targets):
    """Предупреждение, если информативная часть картинки k не в своём углу."""
    for k in range(4):
        inside = targets[k][ROI[k]].astype(float).sum()
        total = targets[k].astype(float).sum() + 1e-9
        if inside / total < 0.9:
            print(f"[run] ВНИМАНИЕ {name}: у изображения {k + 1} только {inside / total:.0%} яркости в его области "
                  f"{[(s.start, s.stop) for s in ROI[k]]}; метод рассчитан на раскладку по углам из задания")


def main():
    p = argparse.ArgumentParser(description="Нейросетевой синтез фазового ДОЭ (4 изображения -> 1 ДОЭ)")
    p.add_argument("--input", default="/data/input", help="папка с комплектами (или с 4 png)")
    p.add_argument("--output", default="/data/output")
    p.add_argument("--mode", default="best", choices=list(MODES))
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--refine", type=int, default=None, help="переопределить число шагов доводки")
    p.add_argument("--threads", type=int, default=0, help="потоков CPU (0 - все)")
    p.add_argument("--no_extra", action="store_true", help="сохранять только doe.png (без восстановлений и метрик)")
    a = p.parse_args()

    if not os.path.isdir(a.input):
        demo = os.path.join(HERE, "examples")
        print(f"[run] папка {a.input} не найдена - запускаю на встроенных примерах {demo}")
        a.input = demo
    device = "cuda" if (a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())) else "cpu"
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    method, refine, restarts, w_frame, norm = MODES[a.mode]
    if a.refine is not None:
        refine = a.refine
    kits = find_kits(a.input)
    if not kits:
        sys.exit(f"[run] в {a.input} не найдено ни одного комплекта из 4 png")
    os.makedirs(a.output, exist_ok=True)
    print(f"[run] комплектов: {len(kits)}, режим: {a.mode}, устройство: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else f" ({torch.get_num_threads()} потоков CPU)"))

    t_load = time.perf_counter()
    model = load_model(a.weights, device)
    with torch.no_grad():                                     # прогрев (первый вызов всегда медленнее)
        model.predict_phase(torch.zeros(1, 4, 128, 128, device=device))
    t_load = time.perf_counter() - t_load

    rows, times = [], {}
    for name, files in kits:
        targets = load_targets(files=files)
        check_layout(name, targets)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        phi, _ = synthesize(targets, method, refine=refine, restarts=restarts, device=device, model=model,
                            verbose=False, w_frame=w_frame, norm=norm)
        doe_u8 = phase_to_uint8(phi)
        dt = time.perf_counter() - t0
        times[name] = dt
        out = os.path.join(a.output, name)
        os.makedirs(out, exist_ok=True)
        if a.no_extra:
            from doe.evaluate import save_gray
            save_gray(os.path.join(out, "doe.png"), doe_u8)
            print(f"[run] {name}: {dt:.2f} с")
            continue
        per, s, _ = evaluate_doe(doe_u8, targets, out_dir=out, norm=norm, verbose=False)
        row = {"kit": name, "ssim": s["ssim_mean"], "pearson": s["pcc_mean"],
               "ssim_frame_norm": s["ssim_frame_mean"], "ssim_roi_fit": s["ssim_fit_mean"],
               "ssim_global": s["ssim_global_mean"], "ssim_matlab": s["ssim_matlab_mean"]}
        if "ssim_skimage_default" in per[0]:
            row["ssim_skimage_default"] = float(np.mean([r["ssim_skimage_default"] for r in per]))
        row["time_sec"] = dt
        rows.append(row)
        print(f"[run] {name:24s} SSIM {row['ssim']:.4f}  Pearson {row['pearson']:.4f}  "
              f"(SSIM при нормировке по кадру {row['ssim_frame_norm']:.4f}, по области {row['ssim_roi_fit']:.4f})  "
              f"время {dt:.2f} с", flush=True)

    total = sum(times.values())
    if rows:
        keys = [k for k in rows[0] if k != "kit"]
        mean = {"kit": "СРЕДНЕЕ", **{k: float(np.mean([r[k] for r in rows])) for k in keys}}
        with open(os.path.join(a.output, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["kit"] + keys)
            w.writeheader()
            w.writerows(rows + [mean])
        print(f"[run] СРЕДНЕЕ по {len(rows)} комплектам: SSIM {mean['ssim']:.4f}, Pearson {mean['pearson']:.4f}, "
              f"время {mean['time_sec']:.2f} с на комплект")
        print(f"[run]   SSIM при нормировке по кадру {mean['ssim_frame_norm']:.4f}, "
              f"при подборе яркости по области {mean['ssim_roi_fit']:.4f}")
        print("[run] SSIM - гауссово окно 11x11 (как skimage gaussian_weights=True); варианты по формуле со слайда, "
              "MATLAB и skimage по умолчанию - в summary.csv")
    with open(os.path.join(a.output, "timing.json"), "w", encoding="utf-8") as f:
        json.dump({"mode": a.mode, "device": device, "model_load_sec": t_load, "per_kit_sec": times,
                   "total_synthesis_sec": total, "mean_per_kit_sec": total / len(times)}, f, indent=2, ensure_ascii=False)
    print(f"[run] загрузка модели {t_load:.1f} с; синтез всех комплектов {total:.1f} с. Результаты: {os.path.abspath(a.output)}")


if __name__ == "__main__":
    main()
