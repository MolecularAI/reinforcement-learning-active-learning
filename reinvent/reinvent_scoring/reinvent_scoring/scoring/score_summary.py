from dataclasses import dataclass
from itertools import compress
import numpy as np
from typing import List

from reinvent_scoring.scoring.component_parameters import ComponentParameters


@dataclass
class ComponentSummary:
    total_score: np.array
    parameters: ComponentParameters
    raw_score: np.ndarray = None
    weight: np.ndarray = None 
    


class FinalSummary:
    def __init__(self, total_score: np.array, scored_smiles: List[str], valid_idxs: List[int],
                 scaffold_log_summary: List[ComponentSummary], total_weight):

        self.total_score = total_score
        self.scored_smiles = scored_smiles
        self.valid_idxs = valid_idxs
        self.total_weight = total_weight

        score = [LoggableComponent(c.parameters.component_type, c.parameters.name, c.total_score) for c in scaffold_log_summary]
        raw_score = [LoggableComponent(c.parameters.component_type, f'raw_{c.parameters.name}', c.raw_score) for c in
                     scaffold_log_summary if c.raw_score is not None]
        weights = [LoggableComponent('Weight', f'Weight_{c.parameters.name}', c.weight) for c in scaffold_log_summary]
        


        score.extend(raw_score)
        score.extend(weights) 
        self.scaffold_log: List[ComponentSummary] = scaffold_log_summary
        self.profile: List[LoggableComponent] = score


    def catch(self, value, *args, **kwargs):
        if value is None:
            return [1] * len(self.scored_smiles)
        else:
            return value 
            


@dataclass
class LoggableComponent:
    component_type: str
    name: str
    score: np.array
