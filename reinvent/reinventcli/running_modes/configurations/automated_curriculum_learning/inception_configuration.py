from dataclasses import dataclass
from typing import List
from pydantic import Field


@dataclass
class InceptionConfiguration:
    memory_size: int = 100
    sample_size: int = 10
    smiles: List[str] = Field(default_factory=list)
    # inputs: List[str] = Field(default_factory=list)

