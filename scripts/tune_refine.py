"""Подбор настроек доводки на примерах обучающего набора (dataset/samples).

    python scripts/tune_refine.py --n 4 --out docs/tuning.md

Сравнивает: шаг Adam (0.02 / 0.05 / 0.1), расписание шага (постоянный, затухание на всём
пути, затухание на последних 30%), разгон шага от ответа сети (0 / 10 шагов), эффект
дискретной доводки 8-битного ДОЭ. Метрика - средний SSIM по информативным областям,
восстановление из 8-битного ДОЭ.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from doe.classic import optimize_phase, polish_quantized  # noqa: E402
from doe.evaluate import evaluate_doe, load_targets  # noqa: E402
from doe.physics import Fresnel, phase_to_uint8  # noqa: E402
from synthesize import load_model  # noqa: E402


def ssim_of(phi, T):
    return evaluate_doe(phase_to_uint8(phi.cpu().numpy()), T, verbose=False)[1]["ssim_mean"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="dataset/samples")
    p.add_argument("--n", type=int, default=4, help="сколько четвёрок взять")
    p.add_argument("--weights", default="weights/doe_unrolled_v2.pt")
    p.add_argument("--out", default="docs/tuning.md")
    a = p.parse_args()
    torch.manual_seed(0)
    kits = sorted(os.listdir(a.samples))[: a.n]
    model = load_model(a.weights, "cpu")
    prop = Fresnel()
    data = []
    for k in kits:
        T = load_targets(os.path.join(a.samples, k))
        t = torch.tensor(T / 255.0, dtype=torch.float32)
        data.append((T, t, model.predict_phase(t[None])[0]))

    def avg(fn):
        return float(np.mean([fn(T, t, p0) for T, t, p0 in data]))

    lines = [f"# Подбор настроек доводки\n", f"Данные: комплектов-четвёрок - {len(kits)}, из `{a.samples}` (обучающий набор Tiny-ImageNet). "
             "SSIM - среднее по 4 изображениям и четвёркам, из 8-битного ДОЭ.\n"]
    t0 = time.time()
    base = avg(lambda T, t, p0: ssim_of(p0, T))
    lines.append(f"Только нейросеть: SSIM **{base:.4f}**\n")

    lines += ["## 1. Шаг Adam (1000 шагов, затухание на последних 30%)\n", "| шаг | SSIM |", "|---|---|"]
    for lr in (0.02, 0.05, 0.1):
        s = avg(lambda T, t, p0: ssim_of(optimize_phase(t, prop, iters=1000, lr=lr, init=p0, cosine=True), T))
        lines.append(f"| {lr} | {s:.4f} |")
        print(lines[-1], flush=True)

    lines += ["", "## 2. Расписание шага (lr 0.05), SSIM после N шагов\n",
              "| расписание | 150 | 300 | 1000 |", "|---|---|---|---|"]
    for name, cos, df in (("постоянный", False, 0.3), ("затухание на всём пути", True, 1.0),
                          ("постоянный + затухание на последних 30%", True, 0.3)):
        cells = []
        for it in (150, 300, 1000):
            cells.append(avg(lambda T, t, p0: ssim_of(optimize_phase(t, prop, iters=it, init=p0, cosine=cos,
                                                                     decay_frac=df), T)))
        lines.append(f"| {name} | " + " | ".join(f"{c:.4f}" for c in cells) + " |")
        print(lines[-1], flush=True)

    lines += ["", "## 3. Разгон шага от ответа сети (lr 0.05), SSIM после N шагов\n",
              "| разгон | 1 | 5 | 10 | 20 |", "|---|---|---|---|---|"]
    for w in (0, 10):
        cells = []
        for it in (1, 5, 10, 20):
            cells.append(avg(lambda T, t, p0: ssim_of(optimize_phase(t, prop, iters=it, init=p0, warmup=w), T)))
        lines.append(f"| {w} шагов | " + " | ".join(f"{c:.4f}" for c in cells) + " |")
        print(lines[-1], flush=True)

    lines += ["", "## 4. Дискретная доводка 8-битного ДОЭ (после 1000 шагов)\n", "| | SSIM |", "|---|---|"]
    res = []
    for T, t, p0 in data:
        q = optimize_phase(t, prop, iters=1000, init=p0, cosine=True)
        res.append((ssim_of(q, T), ssim_of(polish_quantized(q, t, prop), T)))
    lines.append(f"| без доводки 8 бит | {np.mean([r[0] for r in res]):.4f} |")
    lines.append(f"| с доводкой 8 бит | {np.mean([r[1] for r in res]):.4f} |")
    print(lines[-2], lines[-1], flush=True)
    lines.append(f"\nВремя расчёта таблиц: {(time.time() - t0) / 60:.0f} мин (CPU).")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("записан", a.out)


if __name__ == "__main__":
    main()
