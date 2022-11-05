from dataclasses import dataclass


@dataclass()
class TrainingParams:
    model_type: str = "LogisticRegression"
    random_state: int = 44
    n_jobs: int = 4
