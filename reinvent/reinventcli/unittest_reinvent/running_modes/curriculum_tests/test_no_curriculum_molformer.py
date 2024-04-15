import os
import shutil
import unittest

import pytest
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.generative_model import GenerativeModel
from reinvent_scoring import ScoringFunctionNameEnum, ScoringFunctionComponentNameEnum, ComponentParameters, \
    ScoringFunctionParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.enums.diversity_filter_enum import DiversityFilterEnum

from running_modes.automated_curriculum_learning.automated_curriculum_runner import AutomatedCurriculumRunner
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_enum import LearningStrategyEnum
from running_modes.automated_curriculum_learning.logging import AutoCLLogger
from running_modes.configurations import CurriculumLoggerConfiguration, \
    GeneralConfigurationEnvelope
from running_modes.configurations.automated_curriculum_learning.automated_curriculum_learning_input_configuration import \
    AutomatedCurriculumLearningMolformerInputConfiguration
from running_modes.configurations.automated_curriculum_learning.curriculum_strategy_input_configuration import \
    CurriculumStrategyInputConfiguration
from running_modes.configurations.automated_curriculum_learning.prodcution_strategy_input_configuration import \
    ProductionStrategyInputConfiguration
from running_modes.enums.curriculum_strategy_enum import CurriculumStrategyEnum
from running_modes.enums.curriculum_type_enum import CurriculumTypeEnum
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.model_type_enum import ModelTypeEnum
from running_modes.enums.production_strategy_enum import ProductionStrategyEnum
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.utils import set_default_device_cuda
from unittest_reinvent.fixtures.test_config import MAIN_TEST_PATH, PRIOR_PATH, MOLFORMER_PRIOR_PATH
from unittest_reinvent.fixtures.test_data import CELECOXIB


@pytest.mark.integration
class TestNoCurriculumMolformer(unittest.TestCase):

    def setUp(self):
        set_default_device_cuda(dont_use_cuda=True)
        self.cs_enum = CurriculumStrategyEnum()
        self.learning_strategy_enum = LearningStrategyEnum()
        self.ps_enum = ProductionStrategyEnum()
        self.lm_enum = LoggingModeEnum()
        self.run_mode_enum = RunningModeEnum()
        self.sf_enum = ScoringFunctionNameEnum()
        self.sf_component_enum = ScoringFunctionComponentNameEnum()
        self.filter_enum = DiversityFilterEnum()
        self.workfolder = MAIN_TEST_PATH
        self.logging_path = f"{self.workfolder}/log"
        smiles = [CELECOXIB]
        model_type = ModelTypeEnum()
        model_regime = GenerativeModelRegimeEnum()

        automated_cl_parameters, parameters = self._create_configuration(smiles)
        prior_config = ModelConfiguration(model_type.MOLFORMER, model_regime.INFERENCE, MOLFORMER_PRIOR_PATH)
        agent_config = ModelConfiguration(model_type.MOLFORMER, model_regime.INFERENCE, MOLFORMER_PRIOR_PATH)
        prior = GenerativeModel(prior_config)
        agent = GenerativeModel(agent_config)

        self.runner = AutomatedCurriculumRunner(automated_cl_parameters, AutoCLLogger(parameters), prior=prior,
                                                agent=agent)

    def tearDown(self):
        set_default_device_cuda(dont_use_cuda=True)
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _create_configuration(self, smiles):

        curriculum_config = CurriculumStrategyInputConfiguration(name=self.cs_enum.NO_CURRICULUM, input=smiles)

        # Production Phase Configuration
        production_sf_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                            specific_parameters={"smiles": smiles},
                                                            component_type=self.sf_component_enum.TANIMOTO_SIMILARITY))

        production_df = DiversityFilterParameters(self.filter_enum.IDENTICAL_MURCKO_SCAFFOLD, 0.05, 25, 0.4)
        #TODO:
        production_inception = None

        learning_strategy = LearningStrategyConfiguration(parameters={"sigma": 120},
                                                          name=self.learning_strategy_enum.DAP_MOLFORMER)

        production_config = ProductionStrategyInputConfiguration(name=self.ps_enum.MOLFORMER,
                                                                 scoring_function=
                                                            ScoringFunctionParameters(
                                                                name=self.sf_enum.CUSTOM_PRODUCT,
                                                                parameters=[production_sf_parameters]),
                                                                 diversity_filter=production_df,
                                                                 inception=production_inception,
                                                                 retain_inception=False, number_of_steps=3,
                                                                 learning_strategy=learning_strategy)

        automated_cl_parameters = AutomatedCurriculumLearningMolformerInputConfiguration(agent=PRIOR_PATH,
                                                                                curriculum_strategy=curriculum_config,
                                                                                production_strategy=production_config,
                                                                                curriculum_type=CurriculumTypeEnum.AUTOMATED)

        logging = CurriculumLoggerConfiguration(recipient=self.lm_enum.LOCAL,
                                                   logging_path=self.logging_path, result_folder=self.workfolder,
                                                   logging_frequency=0, job_name="unit_test_job")

        parameters = GeneralConfigurationEnvelope(parameters=vars(automated_cl_parameters), logging=vars(logging),
                                                  run_type=self.run_mode_enum.CURRICULUM_LEARNING, version="3.0")

        return automated_cl_parameters, parameters

    def test_automated_curriculum_learning(self):
        self.runner.run()
        self.assertTrue(os.path.isdir(self.logging_path))