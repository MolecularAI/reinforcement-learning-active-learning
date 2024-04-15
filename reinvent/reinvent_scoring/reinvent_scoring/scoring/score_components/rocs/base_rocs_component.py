import glob
import os
import re
import pathlib
from abc import abstractmethod
from typing import List
from openeye import oechem, oeshape
import numpy as np

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_components.rocs.default_values import ROCS_DEFAULT_VALUES
from reinvent_scoring.scoring.score_summary import ComponentSummary
from reinvent_scoring.scoring.enums import ROCSSpecificParametersEnum

class BaseROCSComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.param_names_enum = ROCSSpecificParametersEnum()
        self.file_path = self._specific_param('ROCS_INPUT')
        self.cff_path = self._specific_param("CUSTOM_CFF")

    def calculate_score_for_step(self, molecules: List, step=-1) -> ComponentSummary:
        return self.calculate_score(molecules, step)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        # NOTE: valid_idxs are determined with RDKit not with Open Eye
        valid_smiles = self._chemistry.mols_to_smiles(molecules)
        raw_score = self._calculate_omega_score(valid_smiles, step)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_score = self._transformation_function(raw_score, transform_params)
        score_summary = ComponentSummary(total_score=transformed_score,
                                         parameters=self.parameters,
                                         raw_score=raw_score)
        return score_summary

    def _specific_param(self, key_enum):
        key = self.param_names_enum.__getattribute__(key_enum)
        default = ROCS_DEFAULT_VALUES[key_enum]
        ret = self.parameters.specific_parameters.get(key, default)
        if ret is not None:
            return ret
        raise KeyError(f"specific parameter \'{key}\' was not set")

    @classmethod
    def _setup_reference_molecule_with_shape_query(cls, shape_query):
        cls.prep = oeshape.OEOverlapPrep()
        qry = oeshape.OEShapeQuery()
        overlay = oeshape.OEOverlay()
        if oeshape.OEReadShapeQuery(shape_query, qry):
            overlay.SetupRef(qry)
        cls.rocs_overlay = overlay
        return overlay

    @classmethod
    def _setup_reference_molecule(cls, file_path):
        cls.prep = oeshape.OEOverlapPrep()
        input_stream = oechem.oemolistream()
        input_stream.SetFormat(oechem.OEFormat_SDF)
        input_stream.SetConfTest(oechem.OEAbsoluteConfTest(compTitles=False))
        refmol = oechem.OEMol()
        if input_stream.open(file_path):
            oechem.OEReadMolecule(input_stream, refmol)
        cff = oeshape.OEColorForceField()
        if cff.Init(oeshape.OEColorFFType_ImplicitMillsDean):
            cls.prep.SetColorForceField(cff)
        cls.prep.Prep(refmol)
        overlay = oeshape.OEMultiRefOverlay()
        overlay.SetupRef(refmol)
        cls.rocs_overlay = overlay
        return overlay

    def _prepare_overlay(self):
        file_extension = pathlib.Path(self.file_path).suffix.lower()
        if file_extension not in ['.sdf', '.sq']:
            raise Exception("File extention of the input file is wrong")
        isSdfFile = file_extension == '.sdf'
        overlay_function = BaseROCSComponent._setup_reference_molecule if isSdfFile else BaseROCSComponent._setup_reference_molecule_with_shape_query
        overlay = overlay_function(self.file_path)
        return overlay

    @abstractmethod
    def _calculate_omega_score(self, smiles, step) -> np.array:
        raise NotImplementedError("_calculate_omega_score method is not implemented")
