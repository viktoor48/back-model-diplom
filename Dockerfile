FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей для OpenCV и Postgres
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Команда запуска
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]