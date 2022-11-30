from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


with DAG(
    dag_id="data_generator",
    start_date=days_ago(1),
    schedule_interval="@daily",
) as dag:
    generate_data = DockerOperator(
        image="airflow-data-generator",
        command="/data/raw/{{ ds }} -c 100",
        network_mode="bridge",
        task_id="docker-data-generator",
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
