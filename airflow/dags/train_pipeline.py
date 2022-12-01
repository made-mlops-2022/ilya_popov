from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


with DAG(
    dag_id="train_pipeline",
    start_date=days_ago(7),
    schedule_interval="@weekly",
) as dag:
    preprocess_data = DockerOperator(
        image="airflow-preprocess",
        command="/data/raw/{{ ds }}/ /data/processed/{{ ds }}/",
        network_mode="bridge",
        task_id="docker-preprocess",
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

    split_data = DockerOperator(
        image="airflow-split-data",
        command="/data/processed/{{ ds }}/",
        network_mode="bridge",
        task_id="docker-split-data",
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

    train_model = DockerOperator(
        image="airflow-train-model",
        command="/data/processed/{{ ds }}/ /data/models/{{ ds }}/",
        network_mode="bridge",
        task_id="docker-train-model",
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

    val_model = DockerOperator(
        image="airflow-val-model",
        command="/data/processed/{{ ds }}/ /data/models/{{ ds }}/",
        network_mode="bridge",
        task_id="docker-val-model",
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

    preprocess_data >> split_data >> train_model >> val_model
