from dataclasses import dataclass, field
from typing import List, Dict, Any

from reinvent_models.molformer.enums import SamplingModesEnum


@dataclass
class SampleFromModelConfiguration:
    model_path: str
    output_smiles_path: str
    num_smiles: int = 1024
    batch_size: int = 128
    with_likelihood: bool = True
    sampling_strategy: str = SamplingModesEnum.BEAMSEARCH
    drop_duplicate: bool = True
    input: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
