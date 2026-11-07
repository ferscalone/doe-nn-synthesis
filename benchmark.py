"""Сравнение скорости и качества: нейросеть vs классические итеративные методы.

    python benchmark.py --images examples/kit01_photos --out bench

Строит кривые «качество от времени» и таблицу bench/table.md - это основной
аргумент на защите: во сколько раз нейросеть ускоряет синтез.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from doe.classic import gerchberg_saxton, optimize_phase
from doe.evaluate import evaluate_doe, load_targets
from doe.physics import Fresnel, phase_to_uint8
from synthesize import load_model


def score(phi, targets):
    _, s, _ = evaluate_doe(phase_to_uint8(phi.cpu().numpy()), targets, verbose=False)
    return s


def curve(run, targets, checkpoints):
    """Запускает итеративный метод, засекая время и качество на заданных итерациях."""
    pts = []
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    t_acc = [0.0]
    t_last = [time.perf_counter()]

    def cb(it, phi):
        sync()
        t_acc[0] += time.perf_counter() - t_last[0]
        if it + 1 in checkpoints:
            pts.append({"iter": it + 1, "sec": t_acc[0], **score(phi, targets)})
        t_last[0] = time.perf_counter()

    t_last[0] = time.perf_counter()
    run(cb)
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="examples/kit01_photos")
    p.add_argument("--ckpt", default="weights/doe_unrolled_v2.pt")
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--out", default="bench")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    os.makedirs(a.out, exist_ok=True)
    targets = load_targets(a.images)
    t = torch.tensor(targets / 255.0, dtype=torch.float32, device=device)
    prop = Fresnel().to(device)
    cps = sorted({c for c in (1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000, 2000)
                  if c <= a.iters} | {a.iters})
    res = {}
    gerchberg_saxton(t, prop, 3)          # прогрев (первый вызов медленнее)
    optimize_phase(t, prop, 3)

    print("[bench] Герчберг-Сэкстон ...")
    res["GS"] = curve(lambda cb: gerchberg_saxton(t, prop, a.iters, callback=cb), targets, cps)
    print("[bench] Adam со случайной фазы ...")
    res["Adam (случайный старт)"] = curve(lambda cb: optimize_phase(t, prop, a.iters, callback=cb), targets, cps)
    if os.path.exists(a.ckpt):
        model = load_model(a.ckpt, device)
        model.predict_phase(t[None])
        sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
        sync()
        t0 = time.perf_counter()
        phi_nn = model.predict_phase(t[None])[0]
        sync()
        t_nn = time.perf_counter() - t0
        res["НС (один проход)"] = [{"iter": 0, "sec": t_nn, **score(phi_nn, targets)}]
        print("[bench] НС + доводка ...")
        pts = curve(lambda cb: optimize_phase(t, prop, a.iters, init=phi_nn, callback=cb), targets, cps)
        for q in pts:
            q["sec"] += t_nn
        res["НС + доводка Adam"] = res["НС (один проход)"] + pts
    else:
        print(f"[bench] чекпойнт {a.ckpt} не найден - сравниваю только классические методы")

    with open(os.path.join(a.out, "bench.json"), "w", encoding="utf-8") as f:
        json.dump({"device": device, "results": res}, f, indent=2, ensure_ascii=False)

    # Таблица: сколько времени нужно каждому методу, чтобы достичь порога качества.
    lines = [f"Устройство: {device}\n", "| метод | итераций | время, с | КК (ср.) | ИСС (ср.) |", "|---|---|---|---|---|"]
    for name, pts in res.items():
        for q in pts:
            if q["iter"] in (0, 1, 5, 10, 20, 50, 100, 200, a.iters):
                lines.append(f"| {name} | {q['iter']} | {q['sec']:.3f} | {q['pcc_mean']:.4f} | {q['ssim_mean']:.4f} |")
    thrs = (0.80, 0.90, 0.95, 0.98)
    lines += ["", "Время до достижения порога ИСС (ср.):", "",
              "| метод | " + " | ".join(f"ИСС>={x:.2f}" for x in thrs) + " |", "|---" * (len(thrs) + 1) + "|"]
    for name, pts in res.items():
        cells = []
        for thr in thrs:
            hit = next((q for q in pts if q["ssim_mean"] >= thr), None)
            cells.append(f"{hit['sec']:.3f} с ({hit['iter']} ит.)" if hit else "не достигнут")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if "НС (один проход)" in res:
        nn_pt = res["НС (один проход)"][0]
        gs_eq = next((q for q in res["GS"] if q["ssim_mean"] >= nn_pt["ssim_mean"]), None)
        ad_eq = next((q for q in res["Adam (случайный старт)"] if q["ssim_mean"] >= nn_pt["ssim_mean"]), None)
        lines += ["", f"Один проход нейросети (ИСС {nn_pt['ssim_mean']:.4f}, {nn_pt['sec'] * 1e3:.1f} мс) по качеству равен: "
                  + (f"{gs_eq['iter']} итерациям GS ({gs_eq['sec'] * 1e3:.1f} мс)" if gs_eq else f"более чем {a.iters} итерациям GS")
                  + "; " + (f"{ad_eq['iter']} итерациям Adam ({ad_eq['sec'] * 1e3:.1f} мс)" if ad_eq
                            else f"более чем {a.iters} итерациям Adam") + "."]
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(a.out, "table.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for name, pts in res.items():
            x = [max(q["sec"], 1e-4) for q in pts]
            st = "o" if len(pts) == 1 else "-o"
            ax[0].plot(x, [q["pcc_mean"] for q in pts], st, label=name, ms=4)
            ax[1].plot(x, [q["ssim_mean"] for q in pts], st, label=name, ms=4)
        for a_, nm in zip(ax, ("КК Пирсона (ср.)", "ИСС / SSIM (ср.)")):
            a_.set_xscale("log")
            a_.set_xlabel(f"время синтеза, с ({device})")
            a_.set_ylabel(nm)
            a_.grid(alpha=0.3)
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "speed_vs_quality.png"), dpi=120)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
