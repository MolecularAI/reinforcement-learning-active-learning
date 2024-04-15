"""Scoring component that uses AiZynth for retro synthesis planning."""

import logging
from typing import List

import numpy as np
from rdkit import Chem

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.base_score_component import (
    BaseScoreComponent,
)
from reinvent_scoring.scoring.score_summary import ComponentSummary
from .aizynth_cli_api import run_aizynth
from .parameters import AiZynthParams
from .util import dataclass_from_dict

logger = logging.getLogger(__name__)


def drop_atommap(smi: str) -> str:
    """Drop atom mapping from SMILES string.

    LinkInvent provides atom mappings in the SMILES string
    that are not compatible with AiZynthFinder.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        logger.warning(f"Failed to parse {smi}, keeping as is.")
        smi2 = smi
    else:
        for a in mol.GetAtoms():
            a.SetAtomMapNum(0)
        smi2 = Chem.MolToSmiles(mol)
    return smi2


class BuildingBlockAvailabilityComponent(BaseScoreComponent):
    """AiZynth one-step synthesis building block availability.

    If a molecule can be synthesized using different reactions,
    with different sets of reactants,
    the maximum score is used.
    """

    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        valid_smiles = self._chemistry.mols_to_smiles(molecules)

        clean_smiles = [drop_atommap(smi) for smi in valid_smiles]

        raw_score = self._score_smiles_aizynthfinder(clean_smiles, step)

        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        total_score = self._transformation_function(raw_score, transform_params)

        score_summary = ComponentSummary(
            total_score=total_score, parameters=self.parameters, raw_score=raw_score
        )

        return score_summary

    def _score_smiles_aizynthfinder(self, smiles: List[str], step) -> np.ndarray:

        params = dataclass_from_dict(AiZynthParams, self.parameters.specific_parameters)

        out = run_aizynth(smiles, params, step)

        batch_scores = {}  # Score for every smi in the batch.
        for mol in out["data"]:
            smi = mol["target"]
            trees = mol["trees"]
            tscores = []  # Scores for every tree for this one smi.
            for t in trees:
                scores = t["scores"]
                bb_score = scores["stock availability"]
                reacticlass_score = scores["reaction class membership"]
                numsteps = scores["number of reactions"]

                # Score by number of steps.
                # Hard cut-off in aizynth on num_steps (depth of search tree).
                # We can add "soft" signal to reward fewer steps.
                # Resulting score is "ease of synthesis", not binary synthesizeability.
                # Could even increase num_steps for aizynth config,
                # and add two penalties: smaller below threshold, higher above.
                numsteps_score = 0.98 ** numsteps

                tscore = bb_score * reacticlass_score * numsteps_score
                tscores.append(tscore)

            batch_scores[smi] = max(tscores, default=0)

        ordered_scores = []
        for smi in smiles:
            s = batch_scores.get(smi, None)
            if s is None:
                logger.warning(
                    f"Missing score for {smi}."
                    f" Got scores for the following: {list(batch_scores.keys())}"
                )
                ordered_scores.append(0)  # Default score. Could be NaN as well.
            else:
                ordered_scores.append(s)
        return np.array(ordered_scores)
