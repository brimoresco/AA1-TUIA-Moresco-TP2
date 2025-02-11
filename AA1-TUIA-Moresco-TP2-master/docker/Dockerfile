# Usa una imagen base ligera de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Instalar las dependencias del archivo requirements.txt
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copiar todos los archivos del directorio actual en el contenedor
COPY . .

# Instalar dependencias del sistema que podrían ser necesarias para la compilación
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gcc \
#     g++ \
#     python3-dev \
#     libc6-dev \
#     libffi-dev \
#     libclang-dev \
#     && rm -rf /var/lib/apt/lists/*

# Verifica que gcc y otras herramientas estén instaladas correctamente
# RUN gcc --version
# RUN python3 --version

# Copiar los archivos de inferencia y dependencias al contenedor
# COPY inferencia.py /inferencia.py  

# Crear una carpeta para los binarios del modelo
# RUN mkdir -p /model

# Copiar los binarios del modelo
# COPY ./model/ /model/



# Comando predeterminado para ejecutar el script de inferencia
CMD ["python", "/app/inferencia.py"]
