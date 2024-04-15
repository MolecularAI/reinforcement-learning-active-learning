from collections import namedtuple
import os

import numpy as np
from openeye import oechem, oeomega, oeshape

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.rocs.base_rocs_component import BaseROCSComponent
from reinvent_scoring.scoring.enums import ROCSSimilarityMeasuresEnum
from reinvent_scoring.scoring.enums import ROCSSpecificParametersEnum


class RocsSimilarity(BaseROCSComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.sim_measure_enum = ROCSSimilarityMeasuresEnum()
        self.shape_weight = self.parameters.specific_parameters[self.param_names_enum.SHAPE_WEIGHT]
        self.color_weight = self.parameters.specific_parameters[self.param_names_enum.COLOR_WEIGHT]
        self.sim_func_name_set = self.__get_similarity_name_set()
        cff_path = self.parameters.specific_parameters.get(self.param_names_enum.CUSTOM_CFF, None)
        self.prep = self.__set_prep(cff_path)
        self.overlay = self._prepare_overlay()
        self.omega = self.__setup_omega()
        oechem.OEThrow.SetLevel(10000)

    def _calculate_omega_score(self, smiles, step=-1) -> np.array:
        scores = []
        predicate = getattr(oeshape, self.sim_func_name_set.predicate)()
        for smile in smiles:
            imol = oechem.OEMol()
            best_score = 0.0
            if oechem.OESmilesToMol(imol, smile):
                if self.omega(imol):
                    self.prep.Prep(imol)
                    score = oeshape.OEBestOverlayScore()
                    self.overlay.BestOverlay(score, imol, predicate)
                    best_score_shape = getattr(score, self.sim_func_name_set.shape)()
                    best_score_color = getattr(score, self.sim_func_name_set.color)()
                    best_score_color = correct_color_score(best_score_color)
                    best_score = ((self.shape_weight * best_score_shape) + (
                            self.color_weight * best_score_color)) / (self.shape_weight + self.color_weight)
            scores.append(best_score)
        return np.array(scores)

    def __setup_omega(self):
        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetStrictStereo(False)
        return oeomega.OEOmega(omegaOpts)

    def __get_similarity_name_set(self):
        similarity_collection_name = self.parameters.specific_parameters.get(self.param_names_enum.SIM_MEASURE,
                                                                        self.sim_measure_enum.TANIMOTO)
        similarity_collection = self.__similarity_collection(similarity_collection_name)
        return similarity_collection

    def __similarity_collection(self, sim_measure_type):
        _SIM_FUNC = namedtuple('sim_func', ['shape', 'color', 'predicate'])
        _SIM_DEF_DICT = {
            self.sim_measure_enum.TANIMOTO: _SIM_FUNC('GetTanimoto', 'GetColorTanimoto', 'OEHighestTanimotoCombo'),
            self.sim_measure_enum.REF_TVERSKY: _SIM_FUNC('GetRefTversky', 'GetRefColorTversky',
                                                         'OEHighestRefTverskyCombo'),
            self.sim_measure_enum.FIT_TVERSKY: _SIM_FUNC('GetFitTversky', 'GetFitColorTversky',
                                                         'OEHighestFitTverskyCombo'),
        }
        return _SIM_DEF_DICT.get(sim_measure_type)

    def __set_prep(self, cff_path):
        prep = oeshape.OEOverlapPrep()
        if cff_path is None:
            cff_path = oeshape.OEColorFFType_ImplicitMillsDean
        cff = oeshape.OEColorForceField()
        if cff.Init(cff_path):
            prep.SetColorForceField(cff)
        else:
            raise ValueError("Custom color force field initialisation failed")
        return prep

def correct_color_score(score):
    if score >= 1.0:
        score = 0.9 # or alternative
    return score