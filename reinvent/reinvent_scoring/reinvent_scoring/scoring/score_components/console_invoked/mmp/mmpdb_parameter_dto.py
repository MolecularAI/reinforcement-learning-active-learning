
from pydantic import BaseModel


class MmpdbParameterDTO(BaseModel):
    num_of_cuts: int = 1
    delimiter: str = 'comma'
    max_variable_heavies: int = 40
    max_variable_ratio: float = 0.33