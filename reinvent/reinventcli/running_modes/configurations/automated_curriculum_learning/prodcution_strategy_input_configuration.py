from dataclasses import dataclass
from typing import List

from pydantic import Field
from reinvent_models.molformer.enums import SamplingModesEnum
from reinvent_scoring import ScoringFunctionParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters

from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration
from running_modes.configurations.automated_curriculum_learning.inception_configuration import InceptionConfiguration


@dataclass
class ProductionStrategyInputConfiguration:
    name: str
    learning_strategy: LearningStrategyConfiguration
    scoring_function: ScoringFunctionParameters
    diversity_filter: DiversityFilterParameters
    inception: InceptionConfiguration = None
    retain_inception: bool = True
    input: List[str] = Field(default_factory=list)
    randomize_input: bool = False
    batch_size: int = 64
    learning_rate: float = 0.0001
    sigma: float = 120
    number_of_steps: int = 100
    distance_threshold: float = 100.
    sample_strategy: str = SamplingModesEnum.MULTINOMIAL