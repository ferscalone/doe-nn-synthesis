"""Готовит файл весов для репозитория из чекпойнта обучения.

    python scripts/export_weights.py runs/main/best.pt weights/doe_unrolled_v2.pt

Берутся только сглаженные веса (EMA) и конфигурация сети; веса сохраняются в
половинной точности (float16): файл вдвое меньше (~65 МБ, проходит лимит GitHub
100 МБ), а при загрузке они переводятся обратно в float32. Разница в качестве
на уровне 1e-4 по SSIM (проверено на examples/).
"""
import os
import sys

import torch


def main(src, dst):
    ck = torch.load(src, map_location="cpu")
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}
    out = {"model": sd, "cfg": ck["cfg"], "iter": ck.get("iter"), "val_loss": ck.get("best"), "stored_dtype": "float16"}
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.save(out, dst)
    print(f"[weights] {src} -> {dst}: {os.path.getsize(dst) / 1e6:.1f} МБ, итерация {out['iter']}, "
          f"val loss {out['val_loss']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
