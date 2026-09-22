FROM python:3.13-slim

WORKDIR /app

COPY requirements/requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY app/ app/
COPY src/ src/
COPY config/ config/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]