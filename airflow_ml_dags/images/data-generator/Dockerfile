FROM python:3.10-slim
RUN mkdir /build/
COPY requirements.txt /build/requirements.txt

RUN pip install -r /build/requirements.txt

COPY data_generator.py /data_generator.py

ENTRYPOINT ["python", "data_generator.py"]