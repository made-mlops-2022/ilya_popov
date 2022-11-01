from dataclasses import Field, dataclass


@dataclass()
class TrainingParams:
    model_type: str = Field(default="LogisticRegression")
    random_state: int = Field(default=44)
    n_jobs: int = Field(default=4)
