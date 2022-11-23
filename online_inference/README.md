# Online Inference

## Start instructions

```bash
ONFIG_PATH=online_inference/configs/download_config.yaml uvicorn --app-dir online_inference/ app:app
```

### Docker
- Build image locally
```bash
docker build -t ilya0100/online_inference:latest online_inference/
```
- Pull from hub
```bash
docker pull ilya0100/online_inference
```
- Run container
```bash
docker run --rm -p 8000:8000 ilya0100/online_inference
```

## Оптимизация размера docker image
|REPOSITORY                 | TAG     | SIZE  |
|---------------------------|---------|-------|
|ilya0100/online_inference  | 0.5     | 633MB |
|ilya0100/online_inference  | latest  | 633MB |
|ilya0100/online_inference  | 0.4     | 640MB |
|ilya0100/online_inference  | 0.3     | 754MB |
|ilya0100/online_inference  | 0.1     | 1.55GB|
|ilya0100/online_inference  | 0.2     | 1.55GB|
- Ver 0.2: За основу был взят стандартый образ python:3.10 без оптимизаций размера
- Ver 0.3: Базовый образ заменен на python:3.10-slim
- Ver 0.4: Добавлен флаг --no-cache-dir при утановке зависимостей
- Latest: Базовый образ заменен на python:3.10-slim-buster