"""Экспорт набора данных для экспертов («все реализованные материалы ... наборы данных»).

    python export_dataset.py --data auto --out dataset            # примеры + описание
    python export_dataset.py --data auto --out dataset --full     # + весь банк картинок (.npz)

Что получается в --out:
  DATASET.md            описание: источник, обработка, разбиение, как генерируются примеры
  samples/0001/1..4.png примеры входов сети ровно в формате проверки (128x128, 8 бит, 1 канал)
  samples_preview.png   мозаика из первых примеров
  bank_gray64.npz       (с --full) все исходные картинки 64x64 + индексы отложенной выборки
"""
import argparse
import os

import numpy as np
import torch

from doe.data import BANK, QuadSampler, bank_source, build_bank
from doe.evaluate import save_gray


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="auto")
    p.add_argument("--n_images", type=int, default=110000)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="dataset")
    p.add_argument("--samples", type=int, default=50, help="сколько примеров-четвёрок сохранить как png")
    p.add_argument("--full", action="store_true", help="сохранить весь банк картинок (сотни МБ)")
    p.add_argument("--seed", type=int, default=0, help="тот же seed, что при обучении")
    a = p.parse_args()

    bank = build_bank(a.data, a.n_images, a.data_dir)
    sampler = QuadSampler(bank, "cpu", seed=a.seed)
    n_val = len(sampler.val)
    os.makedirs(os.path.join(a.out, "samples"), exist_ok=True)

    g = torch.Generator().manual_seed(2024)
    quads = (sampler.sample(a.samples, "train", g).numpy() * 255).round().astype(np.uint8)
    for i, q in enumerate(quads):
        d = os.path.join(a.out, "samples", f"{i + 1:04d}")
        os.makedirs(d, exist_ok=True)
        for k in range(4):
            save_gray(os.path.join(d, f"{k + 1}.png"), q[k])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = min(6, len(quads))
        fig, ax = plt.subplots(rows, 4, figsize=(8, 2 * rows))
        for i in range(rows):
            for k in range(4):
                ax[i, k].imshow(quads[i, k], cmap="gray", vmin=0, vmax=255)
                ax[i, k].set_xticks([])
                ax[i, k].set_yticks([])
                if i == 0:
                    ax[i, k].set_title(f"{k + 1}.png")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "samples_preview.png"), dpi=90)
        plt.close(fig)
    except ImportError:
        pass

    if a.full:
        perm = torch.randperm(len(bank), generator=torch.Generator().manual_seed(a.seed)).numpy()
        np.savez_compressed(os.path.join(a.out, "bank_gray64.npz"), bank=bank,
                            val_indices=perm[:n_val], train_indices=perm[n_val:])

    real = bank_source(a.data, a.n_images, a.data_dir)
    descr = {
        "tinyimagenet": "Tiny-ImageNet (HuggingFace [`zh-plus/tiny-imagenet`](https://huggingface.co/datasets/zh-plus/tiny-imagenet), части `train` 100 000 + `valid` 10 000) - "
                        "подмножество ImageNet, 200 классов, цветные фото 64x64; используется для исследовательских "
                        "целей, как и сам ImageNet.",
        "cifar10": "CIFAR-10 (torchvision, 60 000 цветных фото 32x32, увеличены до 64x64 бикубически) - "
                   "запасной источник, если HuggingFace был недоступен.",
        "synthetic": "процедурные картинки (гладкий шум, круги, градиенты), без скачивания - "
                     "только для проверки кода, не для итогового обучения.",
    }.get(real, f"картинки из папки `{real[len('folder:'):]}`" if real.startswith("folder:") else real)
    text = f"""# Набор данных для обучения нейросети синтеза ДОЭ

## Что это
Сети не нужны готовые ДОЭ: она учится через физическую модель (дифракция Френеля),
поэтому набор данных - это только **изображения-цели**.

## Источник
* Фактический источник: **{real}** (параметр `--data {a.data}`); банк - **{len(bank)}** картинок.
* {descr}

## Обработка
1. Каждое фото переводится в оттенки серого (8 бит) и, если нужно, приводится к {BANK}x{BANK}.
2. Разбиение: 2% банка (**{n_val}** картинок) - отложенная выборка для выбора лучшей модели
   (сеть их не видит), остальное - обучение. Перемешивание с seed={a.seed}.
3. Обучающий пример (генерируется на лету, каждый раз новый):
   * 4 случайные картинки банка, случайный фрагмент 62x62;
   * случайные отражения / транспонирование, гамма 0.64-1.57, контраст и уровень чёрного;
   * картинка k кладётся в свой угол чёрного кадра 128x128 - ровно как в проверке:
     1: строки/столбцы [0:62, 0:62], 2: [0:62, 66:128], 3: [66:128, 0:62], 4: [66:128, 66:128].
   Число разных комбинаций практически бесконечно, поэтому переобучения нет.

## Файлы
* `samples/NNNN/1.png ... 4.png` - {a.samples} примеров обучающих входов в формате задания.
* `samples_preview.png` - мозаика из первых примеров.
* `bank_gray64.npz` - {"весь банк (`bank`: uint8 [N,64,64], `val_indices`, `train_indices`)" if a.full else "не сохранён (запустите с `--full`)"}.

## Как воспроизвести
```
python export_dataset.py --data {a.data} --n_images {a.n_images} --out {a.out} {"--full" if a.full else ""}
```
"""
    with open(os.path.join(a.out, "DATASET.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[dataset] готово: {os.path.abspath(a.out)} (банк {bank.shape}, примеров {a.samples})")


if __name__ == "__main__":
    main()
