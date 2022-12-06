from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    target_col: Optional[str]
    scaler: str = "Standart"
    numerical_features: List[str] = field(default_factory=list)
    features_to_drop: List[str] = field(default_factory=list)
