FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    APP_DIR=/app

# Pinned versions so every trial runs on exactly the same libraries.
RUN pip install numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 joblib==1.4.2

WORKDIR /app
COPY data/ /app/data/
COPY pipeline/ /app/pipeline/
RUN mkdir -p /app/artifacts

CMD ["bash"]
