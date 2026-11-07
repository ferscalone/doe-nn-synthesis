"""Собирает независимые тестовые комплекты examples/kitNN/1..4.png.

Комплекты НЕ используются при обучении сети (она училась на Tiny-ImageNet).
Источники: стандартные изображения из библиотеки scikit-image (`skimage.data`,
свободные изображения) и синтетические объекты, нарисованные этим скриптом.
Формат - ровно как в задании: 128x128, 8 бит, оттенки серого; картинка k лежит
в своём углу чёрного кадра, информативная область 62x62:
    1: [0:62, 0:62]   2: [0:62, 66:128]   3: [66:128, 0:62]   4: [66:128, 66:128]

    python scripts/make_example_kits.py            # -> examples/
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from doe.physics import ROI  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
S = 62


def gray(a):
    a = np.asarray(a)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    if a.dtype != np.uint8:
        a = (255 * (a - a.min()) / (a.max() - a.min() + 1e-12)).astype(np.uint8)
    im = Image.fromarray(a)
    return im.convert("L")


def square(im, box=None):
    """Квадратный фрагмент (центр или заданный box в долях) -> 62x62."""
    w, h = im.size
    if box is None:
        s = min(w, h)
        x, y = (w - s) // 2, (h - s) // 2
        im = im.crop((x, y, x + s, y + s))
    else:
        x0, y0, x1, y1 = box
        im = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    return np.asarray(im.resize((S, S), Image.LANCZOS))


def draw(fn):
    im = Image.new("L", (S * 4, S * 4), 0)
    fn(ImageDraw.Draw(im), S * 4)
    return np.asarray(im.resize((S, S), Image.LANCZOS))


def text_img(txt, size):
    im = Image.new("L", (S * 4, S * 4), 15)
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=size)
    bb = d.textbbox((0, 0), txt, font=font)
    d.text(((S * 4 - (bb[2] - bb[0])) / 2 - bb[0], (S * 4 - (bb[3] - bb[1])) / 2 - bb[1]), txt, fill=235, font=font)
    return np.asarray(im.resize((S, S), Image.LANCZOS))


def blobs(seed, thr=0.5):
    rng = np.random.default_rng(seed)
    a = rng.random((16, 16))
    im = Image.fromarray((a * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC)
    a = np.asarray(im) / 255.0
    return ((a > thr) * 230 + 10).astype(np.uint8)


def kits():
    import skimage.data as d
    yy, xx = np.mgrid[0:S, 0:S] / (S - 1)
    r = np.hypot(xx - 0.5, yy - 0.5)
    return {
        "kit01_photos": [square(gray(d.astronaut()), (0.35, 0.05, 0.75, 0.45)), square(gray(d.camera())),
                         square(gray(d.coffee())), square(gray(d.chelsea()))],
        "kit02_textures": [square(gray(d.brick())), square(gray(d.grass())), square(gray(d.gravel())),
                           square(gray(d.coins()))],
        "kit03_text": [text_img("DOE", 110), square(gray(d.text()), (0.0, 0.0, 0.45, 1.0)),
                       square(gray(d.page()), (0.0, 0.0, 0.5, 1.0)), text_img("128", 110)],
        "kit04_shapes": [
            draw(lambda g, n: g.ellipse((n * .2, n * .2, n * .8, n * .8), outline=255, width=n // 14)),
            draw(lambda g, n: g.rectangle((n * .22, n * .22, n * .78, n * .78), fill=220)),
            draw(lambda g, n: g.polygon([(n * .5, n * .1), (n * .9, n * .85), (n * .1, n * .85)], fill=200)),
            draw(lambda g, n: [g.ellipse((n * c - n * .12, n * c2 - n * .12, n * c + n * .12, n * c2 + n * .12), fill=255)
                               for c, c2 in ((.3, .3), (.7, .3), (.3, .7), (.7, .7))])],
        "kit05_binary": [square(gray(d.horse()).point(lambda v: 255 - v)), square(gray(d.shepp_logan_phantom())),
                         square(gray(d.checkerboard())), blobs(7)],
        "kit06_science": [square(gray(d.moon())), square(gray(d.hubble_deep_field()), (0.3, 0.3, 0.6, 0.6)),
                          square(gray(d.immunohistochemistry())), square(gray(d.retina()), (0.25, 0.25, 0.75, 0.75))],
        "kit07_mixed": [square(gray(d.rocket())), square(gray(d.cell())), square(gray(d.microaneurysms())),
                        square(gray(d.clock()))],
        "kit08_smooth": [(xx * 230 + 10).astype(np.uint8), (np.clip(1 - r * 1.6, 0, 1) * 240).astype(np.uint8),
                         ((np.sin(xx * 12) * 0.5 + 0.5) * 200 + 20).astype(np.uint8),
                         (np.clip(square(gray(d.camera())) * 0.25 + 90, 0, 255)).astype(np.uint8)],
    }


def main():
    for name, imgs in kits().items():
        dd = os.path.join(OUT, name)
        os.makedirs(dd, exist_ok=True)
        for k, a in enumerate(imgs):
            frame = np.zeros((128, 128), np.uint8)
            frame[ROI[k]] = np.asarray(a, np.uint8)
            Image.fromarray(frame, mode="L").save(os.path.join(dd, f"{k + 1}.png"))
        print("ok", name)


if __name__ == "__main__":
    main()
