# Образ для запуска синтеза ДОЭ на CPU (работает на любой машине с Docker).
# Сборка:  docker build -t doe-nn .
# Запуск:  docker run --rm -v <папка_с_комплектами>:/data/input -v <папка_результатов>:/data/output doe-nn
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY doe/ doe/
COPY weights/ weights/
COPY examples/ examples/
COPY run.py synthesize.py benchmark.py selfcheck.py ./

# По умолчанию: все комплекты из /data/input -> результаты в /data/output (режим best).
# Если /data/input не подключена, запускается на встроенных примерах examples/.
ENTRYPOINT ["python", "run.py"]
CMD ["--input", "/data/input", "--output", "/data/output"]
