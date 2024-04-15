from abc import ABC, abstractmethod
from typing import List

from reinvent_chemistry.conversions import Conversions

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_summary import ComponentSummary
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_components.active_learning.default_values import (
    ACTIVE_LEARNING_DEFAULT_VALUES,
)
from reinvent_scoring.scoring.enums.active_learning_parameters_enum import (
    active_learning_parameters_enum,
)


class BaseALComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.param_names_enum = active_learning_parameters_enum()

    def calculate_score_for_step(self, molecules: List, step=-1) -> ComponentSummary:
        return self.calculate_score(molecules=molecules, step=step)

    def calculate_score(self, molecules: List, step) -> ComponentSummary:
        # NOTE: valid_idxs are determined with RDKit not with Open Eye
        valid_smiles = self._chemistry.mols_to_smiles(molecules)
        raw_score, weight = self._calculate_AL_scores(valid_smiles, step)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_score = self._transformation_function(raw_score, transform_params)
        score_summary = ComponentSummary(
            total_score=transformed_score,
            parameters=self.parameters,
            raw_score=raw_score,
            weight=weight, 
        )
        return score_summary

    def _specific_param(self, key_enum):
        key = self.param_names_enum.__getattribute__(key_enum)
        default = ACTIVE_LEARNING_DEFAULT_VALUES[key_enum]
        ret = self.parameters.specific_parameters.get(key, default)
        if ret is not None:
            return ret
        raise KeyError(f"specific parameter '{key}' was not set")

    def _calculate_AL_scores(valid_smiles, step):
        raise NotImplementedError
