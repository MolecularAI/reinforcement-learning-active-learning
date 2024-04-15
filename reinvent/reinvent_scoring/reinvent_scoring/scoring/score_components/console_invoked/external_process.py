"""Run an external scoring subprocess

Run external process: provide specific command line parameters when needed
pass on the SMILES as a series of strings at the end.
"""

import os
import subprocess as sp
import shlex
import json
import math
from typing import List

from rdkit import Chem

from reinvent_scoring.scoring.enums import TransformationTypeEnum, TransformationParametersEnum
from reinvent_scoring.scoring.score_transformations import TransformationFactory
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_summary import ComponentSummary
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum


class ExternalProcess:
    """Run an external process for scoring"""

    def __init__(self, parameters: ComponentParameters):
        self.component_specific_parameters = ComponentSpecificParametersEnum()
        self.parameters = parameters
        self._transformation_function = self._assign_transformation(
            self.parameters.specific_parameters
        )

    def calculate_score(self, molecules: List[Chem.Mol], step=-1) -> ComponentSummary:
        """Calculate the scores by running an external process

        The (optional) arguments to the executable are followed by the SMILES
        string by string.  The executable is expected to return the scores
        in a JSON list.  E.g. run a conda script (predict.py) in a specific
        environment (qptuna):

        specific_parameters.executable = "/home/user/miniconda3/condabin/conda"
        specific_parameters.args = "run -n qptuna python predict.py"

        And predict.py as

        import sys
        import pickle
        import json

        smilies = sys.stdin.readlines()

        with open('model.pkl', 'rb') as pf:
            model = pickle.load(pf)

        scores = model.predict_from_smiles(smilies)

        print(json.dumps(list(scores)))

        :param molecules:  RDKit molecules
        :param step: unused, here for compatibility only
        :return: a component summary object
        """

        smilies = [
            Chem.MolToSmiles(molecule) if molecule else ["INVALID"] for molecule in molecules
        ]

        if "executable" not in self.parameters.specific_parameters:
            raise RuntimeError(f"{__name__}: need to provide executable")

        executable = os.path.abspath(self.parameters.specific_parameters["executable"])
        args = self.parameters.specific_parameters.get("args", "")

        sp_args = [executable] + shlex.split(args)
        smiles_input = "\n".join(smilies)

        try:
            result = sp.run(
                sp_args,
                capture_output=True,
                text=True,
                check=True,
                shell=False,
                input=smiles_input
            )
        except sp.CalledProcessError as error:
            ret = error.returncode
            out = error.stdout
            err = error.stderr

            raise RuntimeError(
                f"{__name__}: the external process has failed with exit code {ret}: "
                f"stdout={out}, stderr={err}"
            )

        return self.parse_output(result.stdout)

    calculate_score_for_step = calculate_score

    def parse_output(self, out):
        preds = json.loads(out)
        raw_scores = [val if not math.isnan(val) else 0.0 for val in preds]
        scores = self._apply_transformation(raw_scores)

        score_summary = ComponentSummary(
            total_score=scores, parameters=self.parameters, raw_score=raw_scores
        )

        return score_summary

    def _apply_transformation(self, predicted_scores):
        parameters = self.parameters.specific_parameters
        transform_params = parameters.get(self.component_specific_parameters.TRANSFORMATION)

        if transform_params:
            score = self._transformation_function(predicted_scores, transform_params)
        else:
            score = predicted_scores

        return score

    def _assign_transformation(self, specific_parameters: dict):
        transformation_type = TransformationTypeEnum()
        transform_params = specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION
        )

        if not transform_params:
            specific_parameters[self.component_specific_parameters.TRANSFORMATION] = {
                TransformationParametersEnum.TRANSFORMATION_TYPE: transformation_type.NO_TRANSFORMATION
            }

        factory = TransformationFactory()

        return factory.get_transformation_function(transform_params)
