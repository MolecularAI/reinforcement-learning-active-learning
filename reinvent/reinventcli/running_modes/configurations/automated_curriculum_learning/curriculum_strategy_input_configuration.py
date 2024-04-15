from dataclasses import dataclass
from typing import List

from pydantic import Field
from reinvent_models.molformer.enums import SamplingModesEnum
from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters

from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration
from running_modes.configurations.automated_curriculum_learning.curriculum_objective import CurriculumObjective
from running_modes.configurations.automated_curriculum_learning.inception_configuration import InceptionConfiguration


@dataclass
class CurriculumStrategyInputConfiguration:
    name: str
    learning_strategy: LearningStrategyConfiguration = None
    curriculum_objectives: List[CurriculumObjective] = None
    diversity_filter: DiversityFilterParameters = None
    inception: InceptionConfiguration = None
    max_num_iterations: int = 100
    input: List[str] = Field(default_factory=list)
    randomize_input: bool = False
    batch_size: int = 64
    learning_rate: float = 0.0001
    sigma: float = 120.
    distance_threshold: float = 100.
    sample_strategy: str = SamplingModesEnum.MULTINOMIAL