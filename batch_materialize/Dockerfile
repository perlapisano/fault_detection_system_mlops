FROM bitnamilegacy/spark:3.5

USER root

RUN pip install --no-cache-dir \
    loguru==0.7.2 \
    pandas==2.2.0 \
    pyarrow==14.0.0 \
    redis==5.0.1

WORKDIR /app/batch_materialize
COPY batch_materialize.py .

ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1

CMD ["python", "batch_materialize.py"]
