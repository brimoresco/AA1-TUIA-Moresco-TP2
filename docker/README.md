# Docker para inferencia

### Requisitos
- Docker instalado en tu máquina.

### Construir la imagen
```bash
docker build -t img .
#docker run -it -v "C:/Users/brisa/OneDrive/Desktop/tp2 aa1 rec/docker/model:/app/docker/model" img 


#docker run -it -v "C:/Users/brisa/OneDrive/Desktop/tp2 aa1 rec:/mnt/data" -v "C:/Users/brisa/OneDrive/Desktop/tp2 aa1 rec/docker/model:/app/docker/model" img

#/mnt/data/TEST.csv
