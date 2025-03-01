# Usar una imagen ligera de Python
FROM python:3.12-slim

# Definir el directorio de trabajo dentro del contenedor
WORKDIR /graphrag_demo

# Copiar el archivo de requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código del proyecto
COPY . .
