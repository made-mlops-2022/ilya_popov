from typing import Union
from pydantic import BaseModel, conlist


class ConditionRequest(BaseModel):
    data: list[conlist(Union[int, float], min_items=13, max_items=13)]
    feature_names: conlist(str, min_items=13, max_items=13)


class ConditionResponse(BaseModel):
    condition: int
