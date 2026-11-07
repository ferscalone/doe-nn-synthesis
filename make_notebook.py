"""Собирает самодостаточный ноутбук notebooks/train_kaggle.ipynb: в него вшиты все
исходники проекта, поэтому на Kaggle/Colab достаточно загрузить один файл.
Запуск: python make_notebook.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["doe/__init__.py", "doe/physics.py", "doe/metrics.py", "doe/model.py", "doe/data.py",
         "doe/classic.py", "doe/evaluate.py", "train.py", "synthesize.py", "run.py", "benchmark.py",
         "export_dataset.py", "selfcheck.py", "scripts/make_example_kits.py", "scripts/export_weights.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [md("""
# Обучение нейросети синтеза ДОЭ (развёрнутая сеть v2) — Kaggle / Colab

Этот ноутбук воспроизводит обучение весов `weights/doe_unrolled_v2.pt` из репозитория.
Исходники проекта вшиты в ноутбук, ничего дополнительно загружать не нужно.

**Данные для обучения:** [Tiny-ImageNet на HuggingFace — `zh-plus/tiny-imagenet`](https://huggingface.co/datasets/zh-plus/tiny-imagenet)
(110 000 фото 64×64: части `train` + `valid`). Скачиваются автоматически. Готовые ДОЭ не нужны:
сеть учится через физическую модель дифракции Френеля.

**Kaggle, обучение в фоне:**
1. Справа *Settings*: **Accelerator → GPU T4 x2**, **Internet → On**.
2. **Save Version → Save & Run All (Commit) → Save**. Вкладку можно закрыть.
3. Через ~11 ч: вкладка **Output** → `doe_results.zip` (веса, лог обучения, результаты на примерах, набор данных).

Обучение само подбирает число итераций под `HOURS` и завершается вовремя (лимит сессии Kaggle — 12 ч).
Исходные веса получены именно так: T4 x2, `HOURS = 10.5`, 11 903 итерации, 9,6 ч.
Дообучение поверх готового: *Add Input → Notebook Output →* прошлая версия ноутбука — прошлый `best.pt`
найдётся автоматически (тёплый старт).
"""),
         code("""
import os, sys, glob
ROOT = '/kaggle/working/doe_nn' if os.path.exists('/kaggle/working') else '/content/doe_nn'
DATA_DIR = '/tmp/doe_data'          # кэш датасета не попадает в Output (он большой)
HOURS = 10.5                        # бюджет на обучение, часов (Kaggle: всего 12 ч на сессию)
for d in ('doe', 'scripts'):
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)
os.chdir(ROOT)
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
import subprocess

def run(cmd, strict=True):
    # Запуск команды с выводом в реальном времени. strict=True: при ошибке ноутбук
    # останавливается (не тратим GPU-квоту зря); strict=False: только предупреждение.
    print('$', cmd, flush=True)
    pr = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                          encoding='utf-8', errors='replace')
    for line in pr.stdout:
        print(line, end='', flush=True)
    code = pr.wait()
    if code != 0:
        msg = f'команда завершилась с ошибкой (код {code}): {cmd}'
        if strict:
            raise RuntimeError(msg)
        print('ВНИМАНИЕ:', msg)
    return code == 0

print('Рабочая папка:', ROOT)
!nvidia-smi -L || echo "GPU не найден - включите GPU в настройках среды"
"""),
         md("## 1. Исходники проекта (записываются в файлы)")]

for f in FILES:
    with open(os.path.join(HERE, f), encoding="utf-8") as fh:
        src = fh.read()
    cells.append(code(f"%%writefile {f}\n" + src))

cells.append(md("## 2. Зависимости"))
cells.append(code("!pip -q install datasets scikit-image"))
cells.append(md("""
## 3. Примеры комплектов и быстрая проверка кода (~3 мин)
Примеры (`examples/`) — 8 независимых комплектов из стандартных изображений scikit-image и синтетических
фигур. В обучении они не участвуют; нужны только для самопроверки и демонстрации после обучения.
При ошибке здесь ноутбук остановится.
"""))
cells.append(code("""
run('python scripts/make_example_kits.py')
run('python selfcheck.py')
run(f"python train.py --data synthetic --n_images 2000 --iters 30 --bs 4 --base 8 --stage_base 8 --stages 3 "
    f"--val_every 15 --resume '' --data_dir {DATA_DIR} --out /tmp/smoke")
"""))
cells.append(md("""
## 4. Основное обучение (~`HOURS` часов)
* Каждые 500 итераций печатаются метрики на отложенной части Tiny-ImageNet (2%, сеть их не видит);
  по ним же выбирается лучшая модель `best.pt`.
* `best.pt` / `last.pt` сохраняются по ходу. При сбоях «loss не число» обучение восстанавливается само.
"""))
cells.append(code("""
prev = sorted(glob.glob('/kaggle/input/**/runs/main/best.pt', recursive=True))
INIT = f"--init {prev[-1]}" if prev else ""
print('Тёплый старт:', prev[-1] if prev else 'нет (обучение с нуля)')
ok = run(f"python train.py --data auto --time_budget_h {HOURS} --amp --data_dir {DATA_DIR} --out runs/main {INIT}",
         strict=False)
if not os.path.exists('runs/main/best.pt'):
    raise RuntimeError('обучение не дало best.pt - смотрите лог выше')
if not ok:
    print('ВНИМАНИЕ: обучение завершилось с ошибкой, но best.pt есть - продолжаю с ним')
run('python scripts/export_weights.py runs/main/best.pt weights/doe_unrolled_v2.pt')
"""))
cells.append(md("## 5. Синтез ДОЭ на примерах и сравнение с классическими методами"))
cells.append(code("""
run('python run.py --input examples --output results_examples --mode fast', strict=False)
run('python run.py --input examples --output results_examples_best --mode best', strict=False)
run('python benchmark.py --images examples/kit01_photos --iters 500 --out bench', strict=False)
from IPython.display import Image, display
for f in ('results_examples_best/kit01_photos/overview.png', 'bench/speed_vs_quality.png'):
    if os.path.exists(f):
        display(Image(f))
"""))
cells.append(md("""
## 6. Набор данных для экспертов
Описание и примеры обучающих входов (`dataset/`) идут в архив; весь банк картинок
(`dataset/bank_gray64.npz`, ~300 МБ) остаётся во вкладке Output отдельно.
"""))
cells.append(code("run(f'python export_dataset.py --data auto --data_dir {DATA_DIR} --out dataset --samples 50 --full', strict=False)"))
cells.append(md("## 7. Архив для скачивания (вкладка Output → doe_results.zip)"))
cells.append(code("""
import zipfile
OUT_ZIP = '/kaggle/working/doe_results.zip' if os.path.exists('/kaggle/working') else '/content/doe_results.zip'
include = ['weights/doe_unrolled_v2.pt', 'runs/main/log.csv', 'runs/main/args.json',
           'results_examples', 'results_examples_best', 'bench',
           'dataset/DATASET.md', 'dataset/samples', 'dataset/samples_preview.png']
with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for item in include:
        if os.path.isdir(item):
            for dp, _, fs in os.walk(item):
                for f in fs:
                    z.write(os.path.join(dp, f))
        elif os.path.exists(item):
            z.write(item)
        else:
            print('нет', item)
print(OUT_ZIP, round(os.path.getsize(OUT_ZIP) / 1e6, 1), 'МБ')
try:
    from google.colab import files
    files.download(OUT_ZIP)
except ImportError:
    print('Готово: doe_results.zip во вкладке Output')
"""))

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "notebooks", "train_kaggle.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, ensure_ascii=False, indent=1)
print("записан", out)
