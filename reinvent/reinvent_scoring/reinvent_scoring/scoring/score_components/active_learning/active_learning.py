import numpy as np
import logging

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums import active_learning_parameters_enum
from reinvent_scoring.scoring.score_components.active_learning.base_al_component import (
    BaseALComponent,
)
from reinvent_scoring.scoring.score_components.active_learning.default_values import (
    ACTIVE_LEARNING_DEFAULT_VALUES,
)
from reinvent_scoring.scoring.score_components.active_learning.scripts.retrospectiveReinvent import (
    run_AL,
)


class ActiveLearning(BaseALComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.al_output = self._specific_param("AL_OUTPUT")
        self._set_acquisition_parameters()
        self._set_virtual_library_parameters()
        self._set_training_pool_parameters()
        self._set_model_parameters()
        self._set_random_intervention_parameters()

    def _calculate_AL_scores(self, smiles, step):
        self.step = step
        self.originalSmiles = smiles

        loglevel = "info"
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError("invalid log level: %s" % loglevel)

        logging.basicConfig(
            filename=f"logfile_AL_{self.oracleChoice}.log",
            level=numeric_level,
        )

        scores, weights = run_AL(
            currentEpoch=self.step,
            warmup=self.warmup,
            originalSmiles=self.originalSmiles,
            trainingPoolEpochStart=self.trainingPoolEpochStart,
            trainingPoolLimit=self.trainingPoolLimit,
            trainingPoolMaxCompounds=self.trainingPoolMaxCompounds,
            molecularRepresentationChoice=self.molecularRepresentationChoice,
            oracleChoice=self.oracleChoice,
            confPath=self.confPath,
            propertyName=self.propertyName,
            surrogateModelChoice=self.surrogateModelChoice,
            acquisitionFunctionChoice=self.acquisitionFunctionChoice,
            acquisitionBatchSize=self.acquisitionBatchSize,
            pathToOutDir=self.al_output,
            direction=self.direction,
            virtualLibraryLoops=self.virtualLibraryLoops,
            virtualLibraryAcquisitionBatchSize=self.virtualLibraryAcquisitionBatchSize,
            loopCondition=self.loopCondition,
            predicted_weights=self.predicted_weights,
            noise_level=self.noise_level,
            confidence_cutoff=self.confidence_cutoff,
            relative_fraction=self.relative_fraction,
            drop_duplicate_smiles=self.drop_duplicate_smiles,
            n_lig=self.n_lig,
            num_cpus=self.num_cpus,
            max_docking_calls=self.max_docking_calls,
            subsample = self.subsample,
            mpo_acquisition = self.mpo_acquisition,
            ucb_beta_value= self.ucb_beta_value
        )
        return np.array(scores), np.array(weights)

    def _set_virtual_library_parameters(self):
        self.loopCondition = self._specific_param("LOOP_CONDITION")
        self.virtualLibraryAcquisitionBatchSize = self._specific_param(
            "VIRTUAL_LIBRARY_ACQUISITION_SIZE"
        )
        self.virtualLibraryLoops = self._specific_param("VIRTUAL_LIBRARY_LOOPS")

    def _set_training_pool_parameters(self):
        self.warmup = self._specific_param("WARMUP")
        self.trainingPoolEpochStart = self._specific_param("TRAINING_POOL_START")
        self.trainingPoolLimit = self._specific_param("TRAINING_POOL_EPOCHS_LIMIT")
        self.trainingPoolMaxCompounds = self._specific_param("MAX_COMPOUNDS")
        self.subsample = self._specific_param("SUB_SAMPLE")

    def _set_model_parameters(self):
        self.molecularRepresentationChoice = self._specific_param(
            "MOLECULAR_REPRESENTATION"
        )
        self.oracleChoice = self._specific_param("ORACLE_CHOICE")
        self.confPath = self._specific_param("ICOLOS_CONF_PATH")
        self.propertyName = self._specific_param("TARGET_PROPERTY")
        self.surrogateModelChoice = self._specific_param("SURROGATE_MODEL")

    def _set_acquisition_parameters(self):
        self.acquisitionFunctionChoice = self._specific_param("ACQUISTION_FUNCTION")
        self.acquisitionBatchSize = self._specific_param("ACQUISITION_SIZE")
        self.direction = self._specific_param("DIRECTION")
        self.predicted_weights = self._specific_param("PREDICTED_WEIGHTS")
        self.drop_duplicate_smiles = self._specific_param("DROP_DUPLICATE_SMILES")
        self.n_lig = self._specific_param("N_LIG")
        self.num_cpus = self._specific_param("NUM_CPUS")
        self.relative_fraction = self._specific_param("RELATIVE_FRACTION")
        self.mpo_acquisition = self._specific_param("MPO_ACQUISITION")
        self.ucb_beta_value = self._specific_param("UCB_BETA_VALUE")

    def _set_random_intervention_parameters(self):
        self.noise_level = self._specific_param("NOISE_LEVEL")
        self.confidence_cutoff = self._specific_param("CONFIDENCE_CUTOFF")
        self.max_docking_calls = self._specific_param("MAX_DOCKING_CALLS")
