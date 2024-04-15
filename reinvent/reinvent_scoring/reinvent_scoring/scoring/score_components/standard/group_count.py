from rdkit import Chem
from typing import List
import numpy as np

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary


class GroupCount(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._smarts_pattern = self.parameters.specific_parameters.get(self.component_specific_parameters.SMILES, '')

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score, raw_score = self._evaluate_batch(molecules, self._smarts_pattern)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _evaluate_batch(self, molecules: List, smarts_pattern: str):
        if Chem.MolFromSmarts(smarts_pattern):
            raw_score = [self._substructure_match(molecule, smarts_pattern) for molecule in molecules]
            raw_score = np.array(raw_score)
            transformed_score = self._apply_transformation(raw_score, self.parameters.specific_parameters)
            return transformed_score, raw_score
        else:
            raise IOError('Invalid GroupCount smarts pattern input')

    def _substructure_match(self, molecule, smarts_pattern: str) -> int:
        match_counts = molecule.GetSubstructMatches(Chem.MolFromSmarts(smarts_pattern))
        return len(match_counts)

    def _apply_transformation(self, score, parameters: dict) -> float:
        transform_params = parameters.get(self.component_specific_parameters.TRANSFORMATION)
        if transform_params:
            transformed_score = self._transformation_function(score, transform_params)
        else:
            transformed_score = score
        return transformed_score
