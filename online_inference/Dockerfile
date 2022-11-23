FROM python:3.10-slim-buster

COPY requirements.txt .

RUN pip install --no-cache-dir -U pip \
    pip install --no-cache-dir -r requirements.txt

COPY models/model.pkl online_inference/models/
COPY configs/model_path.yaml online_inference/configs/
COPY entities online_inference/entities/
COPY app.py download_model.py online_inference/

EXPOSE 8000

ENV CONFIG_PATH=online_inference/configs/model_path.yaml

CMD ["uvicorn", "--app-dir", "online_inference/", "--host", "0.0.0.0", "--port", "8000", "app:app"]