"""Обучающие данные.

Готовые ДОЭ («правильные ответы») не нужны: сети достаточно целевых
изображений, а качество оценивается физической моделью. Поэтому датасет -
это просто много разных полутоновых картинок. Из «банка» фрагментов 64x64
на лету собираются четвёрки 62x62 и раскладываются по углам кадра 128x128
так же, как в проверочных изображениях.

Источники банка (--data):
  tinyimagenet  - 100k фото 64x64 с HuggingFace (нужен интернет, ~250 МБ) [по умолч.]
  cifar10       - 60k фото 32x32 через torchvision (увеличиваются до 64x64)
  folder:ПУТЬ   - любая папка с картинками (например, датасет Kaggle)
  synthetic     - процедурные картинки без скачивания (только для проверки кода)
  auto          - tinyimagenet -> cifar10 -> synthetic (первое, что сработает)
"""
import glob
import os

import numpy as np
import torch
from PIL import Image

from .physics import N, ROI, ROI_SIZE

BANK = 64


def _to_gray64(im):
    im = im.convert("L")
    if im.size != (BANK, BANK):
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        im = im.resize((BANK, BANK), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def _bank_tinyimagenet(n_max):
    from datasets import load_dataset
    out = []
    for split in ("train", "valid"):
        ds = load_dataset("zh-plus/tiny-imagenet", split=split)
        for ex in ds:
            out.append(_to_gray64(ex["image"]))
            if len(out) >= n_max:
                return np.stack(out)
    return np.stack(out)


def _bank_cifar10(n_max, root):
    import torchvision
    out = []
    for train in (True, False):
        ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        for im, _ in ds:
            out.append(_to_gray64(im))
            if len(out) >= n_max:
                return np.stack(out)
    return np.stack(out)


def _bank_folder(path, n_max, crops_per_image=8, seed=0):
    rng = np.random.default_rng(seed)
    files = []
    for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"):
        files += glob.glob(os.path.join(path, "**", f"*.{ext}"), recursive=True)
        files += glob.glob(os.path.join(path, "**", f"*.{ext.upper()}"), recursive=True)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"нет картинок в {path}")
    rng.shuffle(files)
    out = []
    for f in files:
        try:
            im = Image.open(f).convert("L")
        except Exception:
            continue
        w, h = im.size
        for _ in range(crops_per_image if min(w, h) > 2 * BANK else 1):
            s = int(rng.integers(BANK, max(BANK + 1, min(w, h) + 1)))
            x, y = int(rng.integers(0, w - s + 1)), int(rng.integers(0, h - s + 1))
            out.append(np.asarray(im.crop((x, y, x + s, y + s)).resize((BANK, BANK), Image.BICUBIC), np.uint8))
        if len(out) >= n_max:
            break
    return np.stack(out[:n_max])


def _bank_synthetic(n_max, seed=0):
    """Процедурные картинки: гладкий шум разных масштабов + фигуры + градиенты."""
    g = torch.Generator().manual_seed(seed)
    out = []
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, BANK), torch.linspace(-1, 1, BANK), indexing="ij")
    bs = 512
    for i in range(0, n_max, bs):
        b = min(bs, n_max - i)
        img = torch.zeros(b, BANK, BANK)
        for s in (4, 8, 16, 32):
            noise = torch.rand(b, 1, s, s, generator=g)
            img += torch.nn.functional.interpolate(noise, size=BANK, mode="bicubic", align_corners=False)[:, 0] * torch.rand(b, 1, 1, generator=g)
        for _ in range(4):
            cx, cy, r = (torch.rand(3, b, 1, 1, generator=g) * torch.tensor([2, 2, 0.6]).view(3, 1, 1, 1) - torch.tensor([1, 1, -0.1]).view(3, 1, 1, 1))
            circ = (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2).float()
            img += circ * (torch.rand(b, 1, 1, generator=g) * 2 - 1)
        ang = torch.rand(b, 1, 1, generator=g) * 6.28
        img += (xx * torch.cos(ang) + yy * torch.sin(ang)) * torch.rand(b, 1, 1, generator=g)
        lo, hi = img.amin((1, 2), keepdim=True), img.amax((1, 2), keepdim=True)
        out.append(((img - lo) / (hi - lo + 1e-6) * 255).byte().numpy())
    return np.concatenate(out)


def bank_source(source="auto", n_max=60000, cache_dir="data"):
    """Какой источник реально использован для банка (после build_bank)."""
    tag = source.replace(":", "_").replace("/", "_").replace("\\", "_")
    path = os.path.join(cache_dir, f"bank_{tag}_{n_max}.npy.source.txt")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return source


def build_bank(source="auto", n_max=60000, cache_dir="data", verbose=True):
    os.makedirs(cache_dir, exist_ok=True)
    tag = source.replace(":", "_").replace("/", "_").replace("\\", "_")
    cache = os.path.join(cache_dir, f"bank_{tag}_{n_max}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    order = ["tinyimagenet", "cifar10", "synthetic"] if source == "auto" else [source]
    last_err = None
    for src in order:
        try:
            if verbose:
                print(f"[data] собираю банк из «{src}» ...", flush=True)
            if src == "tinyimagenet":
                bank = _bank_tinyimagenet(n_max)
            elif src == "cifar10":
                bank = _bank_cifar10(n_max, cache_dir)
            elif src.startswith("folder:"):
                bank = _bank_folder(src[len("folder:"):], n_max)
            elif src == "synthetic":
                bank = _bank_synthetic(n_max)
            else:
                raise ValueError(f"неизвестный источник {src}")
            np.save(cache, bank)
            with open(cache + ".source.txt", "w", encoding="utf-8") as f:
                f.write(src)                     # какой источник реально сработал (для DATASET.md)
            if verbose:
                print(f"[data] банк: {bank.shape}, сохранён в {cache}")
            return bank
        except Exception as e:      # пробуем следующий источник
            last_err = e
            print(f"[data] «{src}» не получилось: {e!r}")
    raise RuntimeError(f"не удалось собрать данные: {last_err!r}")


class QuadSampler:
    """Генератор батчей прямо на GPU: случайные 4 фрагмента -> аугментации ->
    раскладка по углам. Возвращает (B, 4, 128, 128) амплитуды в [0, 1]."""

    def __init__(self, bank_u8, device, val_frac=0.02, seed=0):
        bank = torch.as_tensor(bank_u8)
        n_val = max(64, int(len(bank) * val_frac))
        perm = torch.randperm(len(bank), generator=torch.Generator().manual_seed(seed))
        self.train = bank[perm[n_val:]].to(device)
        self.val = bank[perm[:n_val]].to(device)
        self.device = device
        self.ar = torch.arange(ROI_SIZE, device=device)

    def _patches(self, src, n, g):
        idx = torch.randint(0, len(src), (n,), device=self.device, generator=g)
        oy = torch.randint(0, BANK - ROI_SIZE + 1, (n,), device=self.device, generator=g)
        ox = torch.randint(0, BANK - ROI_SIZE + 1, (n,), device=self.device, generator=g)
        p = src[idx[:, None, None], oy[:, None, None] + self.ar[None, :, None],
                ox[:, None, None] + self.ar[None, None, :]].float() / 255.0
        r = torch.rand(n, 6, device=self.device, generator=g)
        p = torch.where(r[:, 0, None, None] < 0.5, p.flip(-1), p)
        p = torch.where(r[:, 1, None, None] < 0.5, p.flip(-2), p)
        p = torch.where(r[:, 2, None, None] < 0.5, p.transpose(-1, -2), p)
        gamma = torch.exp((r[:, 3] - 0.5) * 0.9)[:, None, None]            # 0.64..1.57
        p = p.clamp(0, 1) ** gamma
        lo = (r[:, 4] * 0.25)[:, None, None]                               # подъём «чёрного»
        hi = (0.6 + 0.4 * r[:, 5])[:, None, None]
        p = lo + (hi - lo) * p
        return p

    def sample(self, batch, split="train", g=None):
        src = self.train if split == "train" else self.val
        p = self._patches(src, batch * 4, g).view(batch, 4, ROI_SIZE, ROI_SIZE)
        out = torch.zeros(batch, 4, N, N, device=self.device)
        for k, (r, c) in enumerate(ROI):
            out[:, k, r, c] = p[:, k]
        return out

    def fixed_val(self, n=128, seed=123):
        g = torch.Generator(device=self.device).manual_seed(seed)
        return self.sample(n, "val", g)
