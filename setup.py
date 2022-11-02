from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="ml_project",
    packages=find_packages(),
    version="0.0.1",
    description="MLOps homework1",
    author="Ilya Popov",
    entry_points={
        "console_scripts": [
            "ml_project_train = ml_project.train_pipeline:train_pipeline_command"
        ]
    },
    install_requires=required,
    license="MIT",
)
