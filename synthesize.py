"""Синтез ДОЭ для 4 изображений и формирование всех результатов для экспертов.

    python synthesize.py --images ПАПКА_С_1-4.png --method best --out results
Для нескольких комплектов сразу удобнее run.py.

Методы (--method):
  nn          - только нейросеть (один проход, миллисекунды)
  nn+refine   - нейросеть + градиентная доводка (--refine шагов)
  best        - максимум качества: доводка с затухающим шагом из ответа сети и из
                --restarts случайных стартов, затем дискретная доводка уже
                8-битного ДОЭ (учёт квантования фазы png); выбирается лучший
                по метрикам; яркость png подбирается под максимум ИСС  [по умолч.]
  adam        - классическая оптимизация фазы с нуля (--iters шагов), для сравнения
  gs          - многоплоскостной Герчберг-Сэкстон (--iters шагов), для сравнения
Результат в --out: doe.png, rec_1..4.png, metrics.json/csv, overview.png, timing.json.
Работает и на CPU.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from doe.classic import gerchberg_saxton, optimize_phase, polish_quantized, timed
from doe.evaluate import evaluate_doe, load_targets
from doe.model import build_model, load_weights
from doe.physics import Fresnel, phase_to_uint8


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ck.get("cfg", {})).to(device).eval()
    load_weights(model, ck["model"])
    return model


def _score(phi, targets_u8, norm, robust=False):
    _, s, _ = evaluate_doe(phase_to_uint8(phi.cpu().numpy()), targets_u8, norm=norm, verbose=False)
    if robust:          # устойчивый режим: лучший тот, у кого выше худший из двух способов нормировки
        return min(s["ssim_fit_mean"], s["ssim_frame_mean"]) + s["pcc_mean"], s
    return s["pcc_mean"] + s["ssim_mean"], s


def synthesize(targets_u8, method="best", ckpt=None, refine=1000, iters=300, lr=0.05, device="cpu",
               w_eff=0.0, restarts=2, norm="best", model=None, verbose=True, w_frame=0.0):
    t = torch.tensor(targets_u8 / 255.0, dtype=torch.float32, device=device)
    prop = Fresnel().to(device)
    timing = {}
    if method in ("nn", "nn+refine", "best"):
        if model is None:
            model = load_model(ckpt, device)
            model.predict_phase(t[None])                  # прогрев
        phi, timing["nn_sec"] = timed(lambda: model.predict_phase(t[None])[0])
        if method == "nn+refine" and refine > 0:
            phi, timing["refine_sec"] = timed(optimize_phase, t, prop, iters=refine, lr=lr, init=phi, w_eff=w_eff,
                                              w_frame=w_frame)
        elif method == "best":
            t0 = time.perf_counter()
            starts = [("нейросеть", phi)] + [(f"случайный {i + 1}", None) for i in range(restarts)]
            best = None
            for j, (name, init) in enumerate(starts):
                cand = optimize_phase(t, prop, iters=refine, lr=lr, init=init, seed=100 + j,
                                      w_eff=w_eff, cosine=True, w_frame=w_frame)
                _, s0 = _score(cand, targets_u8, norm)
                cand = polish_quantized(cand, t, prop, w_eff=w_eff, w_frame=w_frame)
                sc, s = _score(cand, targets_u8, norm, robust=w_frame > 0)
                if verbose:
                    print(f"[best] старт «{name}»: КК {s['pcc_mean']:.5f}  ИСС {s['ssim_mean']:.5f}  "
                          f"(до учёта квантования ИСС {s0['ssim_mean']:.5f})")
                if best is None or sc > best[0]:
                    best = (sc, cand, name)
            phi = best[1]
            timing["refine_sec"] = time.perf_counter() - t0
            timing["chosen_start"] = best[2]
    elif method == "adam":
        phi, timing["adam_sec"] = timed(optimize_phase, t, prop, iters=iters, lr=lr, w_eff=w_eff)
    elif method == "gs":
        phi, timing["gs_sec"] = timed(gerchberg_saxton, t, prop, iters=iters)
    else:
        raise ValueError(method)
    timing["total_sec"] = sum(v for v in timing.values() if isinstance(v, float))
    return phi.cpu().numpy(), timing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="examples/kit01_photos")
    p.add_argument("--files", nargs=4, default=None, help="4 файла вместо папки (по порядку 1..4)")
    p.add_argument("--ckpt", default="weights/doe_unrolled_v2.pt")
    p.add_argument("--method", default="best", choices=["nn", "nn+refine", "best", "adam", "gs"])
    p.add_argument("--refine", type=int, default=2000, help="шагов доводки (nn+refine, best)")
    p.add_argument("--restarts", type=int, default=0, help="best: сколько дополнительных случайных стартов")
    p.add_argument("--iters", type=int, default=300, help="шагов для adam / gs")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--norm", default=None, choices=["fit", "max", "best", "frame"],
                   help="яркость восстановленных png: frame (максимум кадра -> 255), fit (МНК), best (максимум ИСС), max (максимум области). "
                        "По умолчанию frame")
    p.add_argument("--w_eff", type=float, default=0.0, help="штраф за свет вне областей при доводке")
    p.add_argument("--w_frame", type=float, default=0.5,
                   help="вес ошибки при нормировке по всему кадру (устойчивость к нормировке у жюри; run.py: 0.5)")
    p.add_argument("--out", default="results")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    norm = a.norm or ("frame" if a.w_frame > 0 or a.method == "nn" else "best")

    targets = load_targets(a.images, a.files)
    phi, timing = synthesize(targets, a.method, a.ckpt, a.refine, a.iters, a.lr, device, a.w_eff,
                             a.restarts, norm, w_frame=a.w_frame)
    doe_u8 = phase_to_uint8(phi)
    print(f"[synth] метод: {a.method}, устройство: {device}, время: " +
          ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in timing.items()))
    evaluate_doe(doe_u8, targets, out_dir=a.out, norm=norm)
    np.save(os.path.join(a.out, "doe_phase_float.npy"), phi)
    with open(os.path.join(a.out, "timing.json"), "w", encoding="utf-8") as f:
        json.dump({"method": a.method, "device": device, "norm": norm, **timing}, f, indent=2, ensure_ascii=False)
    print(f"[synth] результаты сохранены в {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
