import os
import multiprocessing
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from openeye import oechem, oeomega, oeshape, oequacpac

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums import ROCSSimilarityMeasuresEnum, ROCSSpecificParametersEnum
from reinvent_scoring.scoring.score_components.rocs import oehelper, oefuncs
from reinvent_scoring.scoring.score_components.rocs.base_rocs_component import BaseROCSComponent
from reinvent_scoring.scoring.score_components.rocs.default_values import ROCS_DEFAULT_VALUES


class ParallelRocsSimilarity(BaseROCSComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        avail_cpus = multiprocessing.cpu_count()
        oechem.OEThrow.SetLevel(10000)
        self.sim_measure_enum = ROCSSimilarityMeasuresEnum()

        self.num_cpus = min(avail_cpus, self._specific_param("MAX_CPUS"))
        self._set_omega_parameters()
        self._set_rocs_parameters()
        self.shape_weight = self._specific_param("SHAPE_WEIGHT")
        self.color_weight = self._specific_param("COLOR_WEIGHT")
        self.sim_func_name_set = oefuncs.get_similarity_name_set(parameters, self.param_names_enum,
                                                                 self.sim_measure_enum)

    def _set_omega_parameters(self):
        self.max_confs = self._specific_param("MAX_CONFS")
        self.erange = self._specific_param("EWINDOW")
        self.enum_stereo = self._specific_param("ENUM_STEREO")
        self.max_stereo = self._specific_param("MAX_STEREO")
        if self.max_stereo == 0:
            self.enum_stereo = False
        self.setup_omega(self.erange, self.max_confs)

    def _set_rocs_parameters(self):
        self.save_overlays = self._specific_param("SAVE_ROCS_OVERLAYS")
        if self.save_overlays:
            self.dir_name = self._specific_param("ROCS_OVERLAYS_DIR")
            self.overlay_prefix = self._specific_param("ROCS_OVERLAYS_PREFIX")
            Path(self.dir_name).mkdir(parents=True, exist_ok=True)

        self.protein_file = ""
        self.ligand_file = ""
        self.neg_vol = self._specific_param("NEGATIVE_VOLUME")
        if self.neg_vol:
            self.protein_file = self._specific_param("PROTEIN_NEG_VOL_FILE")
            self.ligand_file = self._specific_param("LIGAND_NEG_VOL_FILE")

    def _calculate_omega_score(self, smiles, step) -> np.array:
        inputs = []
        if len(smiles) == 0:
            return np.array(())
        self._prepare_overlay()
        oefuncs.init_cff(BaseROCSComponent.prep, self.cff_path)
        ind = str(step).zfill(4)

        for smile in smiles:
            input = {"smile": smile, "shape_weight": self.shape_weight, "color_weight": self.color_weight,
                     "sim_func_name_set": self.sim_func_name_set, "batch_id": ind,
                     "enum_stereo": self.enum_stereo, "max_stereo": self.max_stereo,
                     "save_overlays": self.save_overlays,
                     "neg_vol_file": self.protein_file, "neg_vol_lig": self.ligand_file
                     }
            inputs.append(input)
        with Pool(processes=min(self.num_cpus, len(inputs))) as pool:
            results = pool.map(self._unfold, inputs)

        scores = []
        if self.save_overlays:
            overlay_filename = self.overlay_prefix + ind + ".sdf"
            overlay_file_path = os.path.join(self.dir_name, overlay_filename)
            outfs = oechem.oemolostream(overlay_file_path)
        for result in results:
            score, outmol = result
            scores.append(score)
            if self.save_overlays:
                oechem.OEWriteMolecule(outfs, outmol)
        return np.array(scores)

    def _unfold(self, args):
        return self.parallel_scoring(**args)

    @classmethod
    def setup_omega(cls, erange, max_confs):
        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetStrictStereo(False)
        omegaOpts.SetEnergyWindow(erange)
        omegaOpts.SetMaxConfs(max_confs)
        cls.omega = oeomega.OEOmega(omegaOpts)
        return cls.omega

    @classmethod
    def parallel_scoring(cls, smile, shape_weight, color_weight, sim_func_name_set, batch_id, enum_stereo=False,
                         max_stereo=0, save_overlays=False, neg_vol_file="", neg_vol_lig=""):
        predicate = getattr(oeshape, sim_func_name_set.predicate)()
        imol = oechem.OEMol()
        outmol = oechem.OEMol()
        best_score = 0.0
        if oechem.OESmilesToMol(imol, smile):
            oequacpac.OEGetReasonableProtomer(imol)
            omega_success, imol = oehelper.get_omega_confs(imol, cls.omega, enum_stereo, max_stereo)
            if omega_success:
                cls.prep.Prep(imol)
                score = oeshape.OEBestOverlayScore()
                cls.rocs_overlay.BestOverlay(score, imol, predicate)
                outmol = oechem.OEGraphMol(imol.GetConf(oechem.OEHasConfIdx(score.GetFitConfIdx())))
                best_score, best_score_shape, best_score_color, neg_score = oehelper.get_score(outmol, score,
                                                                                               sim_func_name_set,
                                                                                               shape_weight,
                                                                                               color_weight,
                                                                                               neg_vol_file,
                                                                                               neg_vol_lig)
                if save_overlays:
                    oeshape.OERemoveColorAtoms(outmol)
                    oehelper.prep_sdf_file(outmol, score, smile, batch_id, best_score_shape, best_score_color,
                                           neg_score)
        return best_score, outmol
