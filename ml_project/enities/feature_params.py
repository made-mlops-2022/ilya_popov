from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    numerical_features: List[str]
    features_to_drop: Optional[List[str]]
    target_col: Optional[str]
    use_log_trick: bool = True
