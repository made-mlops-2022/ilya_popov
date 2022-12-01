from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from docker.types import Mount


LOCAL_DATA_DIR = Variable.get("LOCAL_DATA_DIR")
MODEL_PATH = Variable.get("MODEL_PATH", default_var="/data/models/{{ ds }}/model.pkl")


with DAG(
    dag_id="predict_pipeline",
    start_date=days_ago(1),
    schedule_interval="@daily",
) as dag:
    preprocess_data = DockerOperator(
        image="airflow-predict",
        command="/data/raw/{{ ds }}/ " + MODEL_PATH + " /data/predictions/{{ ds }}",
        network_mode="bridge",
        task_id="docker-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source=LOCAL_DATA_DIR,
                target="/data",
                type="bind",
            )
        ],
    )
