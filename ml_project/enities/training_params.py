from dataclasses import Field, dataclass


@dataclass()
class TrainingParams:
    model_type: str = Field(default="LogisticRegression")
