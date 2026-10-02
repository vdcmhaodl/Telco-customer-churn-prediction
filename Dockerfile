FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install \
    --no-cache-dir \
    -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/

RUN python -m pip install \
    --no-cache-dir \
    -e .

COPY artifacts/ ./artifacts/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "churn_prediction.api.app:app", "--host", "0.0.0.0", "--port", "8000"]