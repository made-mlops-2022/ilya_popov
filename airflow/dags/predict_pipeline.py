from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


with DAG(
    dag_id="predict_pipeline",
    start_date=days_ago(1),
    schedule_interval="@daily",
) as dag:
    preprocess_data = DockerOperator(
        image="airflow-predict",
        command="/data/raw/{{ ds }}/ /data/models/{{ ds }}/ /data/predictions/{{ ds }}",
        network_mode="bridge",
        task_id="docker-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source="/home/ilya0100/documents/MLOps/ilya_popov/airflow/data/",
                target="/data",
                type="bind",
            )
        ],
    )
