from os import devnull
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import List

import chemprop
from rdkit import Chem

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary
from reinvent_scoring.scoring.score_transformations import TransformationFactory
from reinvent_scoring.scoring.enums import (
    TransformationTypeEnum,
    TransformationParametersEnum,
)


@contextmanager
def suppress_output():
    """Context manager to redirect stdout and stderr to /dev/null"""

    with open(devnull, "w") as nowhere:
        with redirect_stderr(nowhere) as err, redirect_stdout(nowhere) as out:
            yield (err, out)


class ChemPropComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

        args = [
            "--checkpoint_dir",  # ChemProp models directory
            parameters.specific_parameters["checkpoint_dir"],
            "--test_path",  # required
            "/dev/null",
            "--preds_path",  # required
            "/dev/null",
        ]

        rdkit2d = parameters.specific_parameters.get("rdkit_2d_normalized", False)

        if rdkit2d:
            args.extend(["--features_generator", "rdkit_2d_normalized", "--no_features_scaling"])

        with suppress_output():
            self.chemprop_args = chemprop.args.PredictArgs().parse_args(args)
            self.chemprop_model = chemprop.train.load_model(args=self.chemprop_args)

        self._transformation_function = self._assign_transformation(parameters.specific_parameters)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        smilies = [
            [Chem.MolToSmiles(molecule)] if molecule else ["INVALID"] for molecule in molecules
        ]

        with suppress_output():
            preds = chemprop.train.make_predictions(
                model_objects=self.chemprop_model,
                smiles=smilies,
                args=self.chemprop_args,
                return_invalid_smiles=True,
                return_uncertainty=False,
            )

        raw_scores = [
            val[0] if not "Invalid SMILES" in val else 0.0 for val in preds
        ]  # FIXME: score 0.0
        scores = self._apply_transformation(raw_scores, self.parameters.specific_parameters)

        score_summary = ComponentSummary(
            total_score=scores, parameters=self.parameters, raw_score=raw_scores
        )

        return score_summary

    def _apply_transformation(self, predicted_activity, parameters: dict):
        transform_params = parameters.get(self.component_specific_parameters.TRANSFORMATION)

        if transform_params:
            activity = self._transformation_function(predicted_activity, transform_params)
        else:
            activity = predicted_activity

        return activity

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


if __name__ == "__main__":
    from rdkit import Chem

    cp = ComponentParameters(
        "",
        "",
        1.0,
        {
            "checkpoint_dir": "3CLPro_6w63",
            "rdkit_2d_normalized": True,
            "transformation": {
                "transformation_type": "reverse_sigmoid",
                "high": -5.0,
                "low": -35.0,
                "k": 0.2,
            },
        },
    )

    component = ChemPropComponent(cp)

    smilies = [
        "Cc1cc(cc(c1O)C)CNC(=O)CSc2ccc(cn2)S(=O)(=O)N3CC[NH+](CC3)Cc4ccccc4",  # -29.21
        "c1ccc-2c(c1)Cc3c2cc(cc3)NC(=O)c4cc(ccc4Br)F",  # -21.10
        "Cn1cc(c(n1)C(F)(F)F)S(=O)(=O)N",  # -13.35
        "CN1C[C@](SC1=S)([C@H]([C@@H]([C@@H](CO)O)O)O)O",  # -8.63
        "INVALID",  # 0.0
        "Cc1c(cn(n1)C)C(=O)N2[C@@H]3CCCC[C@@H]3C[C@H]2C(=O)[O-]",
    ]  # -11.14
    mols = [Chem.MolFromSmiles(smiles) for smiles in smilies]
    score_summary = component.calculate_score(mols)
    print(score_summary.total_score)
    print(score_summary.raw_score)
