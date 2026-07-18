FROM bitnamilegacy/spark:3.5

USER root

RUN pip install --no-cache-dir \
    loguru==0.7.2 \
    pandas==2.2.0 \
    pyarrow==14.0.0

WORKDIR /app

COPY batch_processing.py .

COPY --from=parent feature_generator.py .

ENV DATA_DIR=/app/data
ENV OUTPUT_DIR=/app/data/features
ENV PYTHONUNBUFFERED=1

CMD ["python", "batch_processing.py"]
