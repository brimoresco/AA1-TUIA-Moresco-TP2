# Proyecto MLOps - Deployment con Docker

## Instrucciones para crear y ejecutar la imagen Docker

1. Clona este repositorio en tu máquina local.

2. Accede a la carpeta `docker` dentro del repositorio.

3. Construye la imagen de Docker:

```bash
docker build -t mlops-inferencia .


#docker run -p 5000:5000 mlops-inferencia
