# Usamos la imagen oficial de Python ligera
FROM python:3.11-slim

# Definimos la carpeta de trabajo
WORKDIR /app

# --- EL PARCHE MÁGICO ---
# Instalamos la librería de C++ que XGBoost necesita para no crashear
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copiamos requerimientos e instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código y modelos
COPY . .

# Forzamos el puerto para Google Cloud Run
ENV PORT=8080
EXPOSE $PORT

# Comando de arranque estricto
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]