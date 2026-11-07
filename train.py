"""Обучение нейросети синтеза ДОЭ.

Основной режим (Kaggle/Colab, GPU; развёрнутая сеть v2, сам подстраивает число
итераций под бюджет времени и сохраняется по ходу):
    python train.py --data auto --time_budget_h 10.5 --amp --out runs/main
Быстрая проверка кода на слабом ПК (CPU, 1-3 минуты):
    python train.py --data synthetic --n_images 2000 --iters 30 --bs 2 --base 8 --stage_base 8 --stages 3 --val_every 15 --out runs/smoke
Старая сеть (v1):  --arch v1

Качество в процессе обучения контролируется только на отложенной части
обучающего набора (2% картинок, сеть их не видит); по ней же выбирается best.pt.
"""
import argparse
import copy
import csv
import json
import math
import os
import time

import torch

import torch.nn as nn

from doe.data import QuadSampler, build_bank
from doe.metrics import full_loss
from doe.model import build_model, load_weights
from doe.physics import ROI, roi_crop

T_START = time.time()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="auto", help="auto | tinyimagenet | cifar10 | folder:ПУТЬ | synthetic")
    p.add_argument("--n_images", type=int, default=110000, help="размер банка картинок")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="runs/main")
    # архитектура
    p.add_argument("--arch", default="v2", choices=["v1", "v2"])
    p.add_argument("--base", type=int, default=32, help="ширина начальной U-Net A")
    p.add_argument("--base_ref", type=int, default=24, help="v1: ширина U-Net B")
    p.add_argument("--stages", type=int, default=12, help="v2: число развёрнутых блоков с физикой")
    p.add_argument("--stage_base", type=int, default=32, help="v2: ширина U-Net в каждом блоке")
    p.add_argument("--depth", type=int, default=3, help="v2: глубина U-Net блоков")
    p.add_argument("--no_grad_ckpt", action="store_true", help="v2: не экономить память (быстрее, но больше VRAM)")
    # обучение
    p.add_argument("--iters", type=int, default=200000, help="максимум итераций (обычно ограничивает бюджет времени)")
    p.add_argument("--time_budget_h", type=float, default=0.0,
                   help="сколько часов можно учиться; число итераций подбирается автоматически (0 - выкл.)")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999, help="скользящее среднее весов (0 - выкл.)")
    p.add_argument("--aux", type=float, default=0.3, help="вес потерь на промежуточных ответах")
    p.add_argument("--w_eff", type=float, default=0.0, help="штраф за свет вне информативных областей")
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--amp", action="store_true", help="смешанная точность для свёрток (быстрее на GPU)")
    p.add_argument("--resume", default="auto",
                   help="продолжить прерванное обучение: путь к last.pt; 'auto' - взять OUT/last.pt, если есть; '' - нет")
    p.add_argument("--init", default="", help="тёплый старт: взять только веса из чекпойнта (новое расписание)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug_nan_at", type=int, default=0, help=argparse.SUPPRESS)   # для проверки восстановления
    return p.parse_args()


class _Phases(nn.Module):
    """Обёртка: forward() возвращает список фаз всех блоков (для DataParallel)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, t):
        return self.m.forward_phases(t)


def model_cfg(a):
    if a.arch == "v1":
        return {"arch": "v1", "base": a.base, "base_ref": a.base_ref}
    return {"arch": "v2", "base": a.base, "stage_base": a.stage_base, "stages": a.stages, "depth": a.depth,
            "grad_ckpt": not a.no_grad_ckpt}


def supervised_loss(model, outs, tgt_roi, aux, w_eff):
    """Глубокий контроль: итоговый ответ с весом 1, промежуточные - с весом
    до aux (линейно растёт к концу). Это помогает каждому блоку учиться."""
    L = len(outs)
    loss, m_final = 0.0, None
    for i, p in enumerate(outs):
        w = 1.0 if i == L - 1 else aux * (i + 1) / (L - 1)
        if w == 0:
            continue
        l, m = full_loss(model.amplitudes(p), tgt_roi, ROI, w_eff)
        loss = loss + w * l
        if i == L - 1:
            m_final = m
    return loss, m_final


@torch.no_grad()
def validate(model, val, w_eff):
    was_training = model.training
    model.eval()
    tot = {"loss": 0.0, "pcc": 0.0, "ssim": 0.0, "pcc0": 0.0}
    nb = 0
    for i in range(0, len(val), 32):
        t = val[i:i + 32]
        outs = model.forward_all(t)
        l, m = full_loss(model.amplitudes(outs[-1]), roi_crop(t), ROI, w_eff)
        _, m0 = full_loss(model.amplitudes(outs[0]), roi_crop(t), ROI, 0.0)
        tot["loss"] += l.item()
        tot["pcc"] += m["pcc"]
        tot["ssim"] += m["ssim"]
        tot["pcc0"] += m0["pcc"]
        nb += 1
    model.train(was_training)
    return {k: v / nb for k, v in tot.items()}


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[train] устройство: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    bank = build_bank(a.data, a.n_images, a.data_dir)
    sampler = QuadSampler(bank, device, seed=a.seed)
    val = sampler.fixed_val(128)

    cfg = model_cfg(a)
    model = build_model(cfg).to(device)
    print(f"[train] архитектура {a.arch}, параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    plan = {"total": a.iters}       # общее число итераций; может уточниться по бюджету времени

    def lr_lambda(i):
        warm = min(1.0, (i + 1) / max(1, a.warmup))
        frac = min(i, plan["total"]) / max(1, plan["total"])
        return warm * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = a.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = copy.deepcopy(model).eval() if a.ema > 0 else model
    if ema is not model:                     # (при --ema 0 веса самой модели не замораживаем)
        for p_ in ema.parameters():
            p_.requires_grad_(False)

    start, best = 0, float("inf")
    resume = os.path.join(a.out, "last.pt") if a.resume == "auto" else a.resume
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device)
        load_weights(model, ck.get("model_raw", ck["model"]))
        if ema is not model:
            load_weights(ema, ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["iter"], ck.get("best", best)
        plan["total"] = ck.get("total", plan["total"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=start - 1)
        print(f"[train] продолжаю {resume} с итерации {start} из {plan['total']}")
    elif a.init:
        ck = torch.load(a.init, map_location=device)
        old = dict(ck.get("cfg", {}))
        old.setdefault("arch", "v1")
        same = all(old.get(k) == v for k, v in cfg.items() if k != "grad_ckpt")
        if not same:
            print(f"[train] ВНИМАНИЕ: {a.init} - другая архитектура ({old}), тёплый старт пропущен, учу с нуля")
        else:
            load_weights(model, ck["model"])
            if ema is not model:
                load_weights(ema, ck["model"])
            print(f"[train] тёплый старт с весов {a.init} (итерация {ck.get('iter', '?')})")

    log_path = os.path.join(a.out, "log.csv")
    new_log = not os.path.exists(log_path) or start == 0
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["iter", "train_loss", "train_pcc", "train_ssim", "val_loss", "val_pcc", "val_ssim",
                      "val_pcc_init", "lr", "sec"])

    def save(it, path, full=True):
        """full=True - всё для продолжения (last.pt); False - только веса для работы (best.pt)."""
        ck = {"model": ema.state_dict(), "iter": it, "best": best, "cfg": cfg}
        if full:
            ck.update({"model_raw": model.state_dict(), "opt": opt.state_dict(), "total": plan["total"]})
        torch.save(ck, path + ".tmp")
        os.replace(path + ".tmp", path)      # атомарно: файл не испортится при обрыве

    # Несколько GPU (на Kaggle «T4 x2»): батч делится между картами.
    runner = _Phases(model)
    n_gpu = torch.cuda.device_count() if device == "cuda" else 0
    if n_gpu > 1:
        try:                                 # пробная итерация: если что-то не так - одна карта
            dp = nn.DataParallel(runner)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                ph = dp(sampler.sample(a.bs))
            sum(f.float().mean() for f in ph).backward()
            opt.zero_grad(set_to_none=True)
            runner = dp
            print(f"[train] использую {n_gpu} GPU (DataParallel), по {a.bs // n_gpu} примеров на карту")
        except Exception as e:  # noqa: BLE001
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"[train] несколько GPU не сработали ({type(e).__name__}: {e}); учусь на одной")

    t_loop = time.time()
    run = {"loss": 0.0, "pcc": 0.0, "ssim": 0.0}
    nrun = 0
    it = start
    measure_from = None
    nan_streak, nan_total, recoveries = 0, 0, 0
    while it < plan["total"]:
        if a.time_budget_h > 0:
            if measure_from is None and it - start == 20:
                measure_from = (it, time.time())
            if measure_from is not None and it - measure_from[0] == 200:
                spi = (time.time() - measure_from[1]) / 200
                left = a.time_budget_h * 3600 - (time.time() - T_START) - 600     # 10 мин запас на финал
                new_total = min(a.iters, it + int(left / (spi * 1.08)))        # +8% на валидации
                plan["total"] = max(new_total, it + 1)
                print(f"[train] {spi:.3f} с/итер. -> всего {plan['total']} итераций "
                      f"(~{(plan['total'] - it) * spi * 1.08 / 3600:.1f} ч)", flush=True)
            if time.time() - T_START > a.time_budget_h * 3600 - 300:
                print("[train] бюджет времени исчерпан, завершаю")
                break
        tgt = sampler.sample(a.bs)
        tr = roi_crop(tgt)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            phis = runner(tgt)
        outs = [torch.polar(torch.ones_like(f), f.float()) for f in phis]
        loss, m = supervised_loss(model, outs, tr, a.aux, a.w_eff)
        it += 1
        if a.debug_nan_at and a.debug_nan_at <= it < a.debug_nan_at + 15 and recoveries == 0:
            loss = loss * float("nan")
        if not torch.isfinite(loss):
            # Обычно причина - переполнение половинной точности (fp16, --amp). Одиночный
            # пропуск безвреден. Если подряд много - веса больше не учатся: восстанавливаемся.
            nan_streak += 1
            nan_total += 1
            if nan_streak <= 3 or nan_streak % 50 == 0:
                print(f"[train] итерация {it}: loss не число, пропускаю шаг (подряд {nan_streak}, всего {nan_total})",
                      flush=True)
            opt.zero_grad(set_to_none=True)
            sched.step()
            if nan_streak >= 10:
                recoveries += 1
                bad = any(not torch.isfinite(q).all() for q in model.parameters())
                with torch.no_grad():          # сглаженная копия весов (EMA) не пострадала
                    for pm, pe in zip(model.parameters(), ema.parameters()):
                        pm.copy_(pe)
                opt.state.clear()              # сбросить накопленные моменты Adam
                what = "веса восстановлены из сглаженной копии (EMA)" + (" - они были испорчены" if bad else "")
                if use_amp:
                    use_amp = False
                    scaler = torch.amp.GradScaler("cuda", enabled=False)
                    what += "; смешанная точность выключена, дальше fp32 (медленнее, но без переполнений)"
                else:
                    for i_ in range(len(sched.base_lrs)):
                        sched.base_lrs[i_] *= 0.5
                    what += "; шаг обучения уменьшен вдвое"
                print(f"[train] {nan_streak} шагов подряд loss не число -> {what}", flush=True)
                if a.time_budget_h > 0:
                    measure_from = (it, time.time())   # скорость изменилась - пересчитать план
                nan_streak = 0
            continue
        nan_streak = 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if ema is not model:
            d = min(a.ema, (1 + it) / (10 + it))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        run["loss"] += loss.item()
        run["pcc"] += m["pcc"]
        run["ssim"] += m["ssim"]
        nrun += 1

        if it % a.val_every == 0 or it == plan["total"]:
            v = validate(ema, val, a.w_eff)
            tr_m = {k: x / max(1, nrun) for k, x in run.items()}
            el = time.time() - t_loop
            print(f"[{it:>6}/{plan['total']}] loss {tr_m['loss']:.4f} | val КК {v['pcc']:.4f} ИСС {v['ssim']:.4f} "
                  f"(старт КК {v['pcc0']:.3f}) | lr {sched.get_last_lr()[0]:.1e} | {el / 60:.1f} мин", flush=True)
            log.writerow([it, tr_m["loss"], tr_m["pcc"], tr_m["ssim"], v["loss"], v["pcc"], v["ssim"], v["pcc0"],
                          sched.get_last_lr()[0], round(el, 1)])
            log_f.flush()
            run = {k: 0.0 for k in run}
            nrun = 0
            if v["loss"] < best:       # выбор модели - по отложенной выборке
                best = v["loss"]
                save(it, os.path.join(a.out, "best.pt"), full=False)
            save(it, os.path.join(a.out, "last.pt"))
    save(it, os.path.join(a.out, "last.pt"))
    log_f.close()
    best_path = os.path.join(a.out, "best.pt")
    if not os.path.exists(best_path):
        save(it, best_path, full=False)
    ck = torch.load(best_path, map_location=device)
    load_weights(ema, ck["model"])
    v = validate(ema, val, a.w_eff)
    print(f"[train] лучшая модель (итерация {ck['iter']}) на отложенной выборке, без доводки: "
          f"КК {v['pcc']:.4f}, ИСС {v['ssim']:.4f}")
    print(f"[train] готово за {(time.time() - T_START) / 3600:.2f} ч. Чекпойнт: {best_path}")


if __name__ == "__main__":
    main()
