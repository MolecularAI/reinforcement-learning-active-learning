import os
import shutil
import unittest
import pytest

from running_modes.configurations import GeneralConfigurationEnvelope
from running_modes.configurations.logging.create_model_log_configuration import CreateModelLoggerConfiguration
from running_modes.create_model.logging.local_create_model_logger import LocalCreateModelLogger
from running_modes.enums.running_mode_enum import RunningModeEnum
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.model_type_enum import ModelTypeEnum

# Model specific imports
from running_modes.create_model import MolformerCreateModelRunner
from running_modes.configurations import MolformerCreateModelConfiguration
from reinvent_models.molformer.dto.molformer_model_parameters_dto import MolformerNetworkParameters

from unittest_reinvent.fixtures.test_config import MAIN_TEST_PATH, MOLFORMER_SMILES_SET_PATH


@pytest.mark.integration
class TestMolformerCreateModel(unittest.TestCase):

    def setUp(self):
        self.rm_enums = RunningModeEnum()
        self.lm_enum = LoggingModeEnum()
        self.mt_enum = ModelTypeEnum()
        self.workfolder = MAIN_TEST_PATH
        self.output_file = os.path.join(self.workfolder, "new_model.empty")
        os.makedirs(self.workfolder, exist_ok=True)

        self.network_parameters = MolformerNetworkParameters()
        self.config = MolformerCreateModelConfiguration(input_smiles_path=MOLFORMER_SMILES_SET_PATH,
                                                        output_model_path=self.output_file,
                                                        network=self.network_parameters)



        log_conf = CreateModelLoggerConfiguration(recipient=self.lm_enum.LOCAL,
                                                  logging_path=os.path.join(MAIN_TEST_PATH, "log"),
                                                  job_name="molformer_create_model_test",
                                                  job_id="1")
        self.configuration_envelope = GeneralConfigurationEnvelope(parameters=vars(self.config),
                                                                   logging=vars(log_conf),
                                                                   run_type=self.rm_enums.CREATE_MODEL,
                                                                   model_type=self.mt_enum.MOLFORMER,
                                                                   version="3.0")
        self.logger = LocalCreateModelLogger(self.configuration_envelope)

        runner = MolformerCreateModelRunner(self.config, self.logger)
        self.model = runner.run()

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def test_save_model(self):
        self.assertTrue(os.path.isfile(self.config.output_model_path))

    def test_correct_model_properties(self):
        self.assertEqual(self.model.max_sequence_length, self.config.max_sequence_length)
        self.assertEqual(len(self.model.vocabulary.tokens()), 20)

    def test_special_tokens(self):
        # token 0 -> '*' is the padding symbol
        # token 1 -> '^' is the start symbol
        # token 2 -> '$' is the end symbol
        self.assertTrue(self.model.vocabulary.decode([0, 1, 2]) == ['*', '^', '$'])
