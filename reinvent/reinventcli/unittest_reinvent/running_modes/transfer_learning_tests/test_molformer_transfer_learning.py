import shutil
import unittest
import os
import pytest

from running_modes.configurations import TransferLearningLoggerConfig, GeneralConfigurationEnvelope
from running_modes.configurations.transfer_learning.molformer_transfer_learning_configuration import \
    MolformerTransferLearningConfiguration
from running_modes.configurations.transfer_learning.noamopt_configuration import NoamoptConfiguration

from running_modes.constructors.transfer_learning_mode_constructor import TransferLearningModeConstructor
from running_modes.utils import set_default_device_cuda
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.running_mode_enum import RunningModeEnum
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum

from unittest_reinvent.fixtures.test_config import MAIN_TEST_PATH, MOLFORMER_PRIOR_PATH, MOLFORMER_SMILES_SET_PATH
from unittest_reinvent.fixtures.utils import count_empty_files




@pytest.mark.integration
class TestMolformerTransferLearning(unittest.TestCase):

    def setUp(self):
        set_default_device_cuda()
        lm_enum = LoggingModeEnum()
        rm_enum = RunningModeEnum()
        mt_enum = ModelTypeEnum()

        self.workfolder = os.path.join(MAIN_TEST_PATH, mt_enum.MOLFORMER + rm_enum.TRANSFER_LEARNING)
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        self.log_dir = os.path.join(self.workfolder, "test_log")
        log_config = TransferLearningLoggerConfig(logging_path=self.log_dir, recipient=lm_enum.LOCAL,
                                                  job_name="test_job")
        self.opt_parameters = NoamoptConfiguration()
        pair_config = {
          'type': 'tanimoto',
          'min_cardinality': 1,
          'max_cardinality': 100,
          'lower_threshold': 0.7,
          'upper_threshold': 1.0
        }
        self.parameters = MolformerTransferLearningConfiguration(input_model_path=MOLFORMER_PRIOR_PATH,
                                                                  output_model_path=os.path.join(self.workfolder,'model.chpt'),
                                                                  input_smiles_path=MOLFORMER_SMILES_SET_PATH,
                                                                  num_epochs=2,
                                                                  optimizer=self.opt_parameters,
                                                                  validation_percentage=0.1,
                                                                  pairs=pair_config,
                                                                  validation_seed=1234)
        self.general_config = GeneralConfigurationEnvelope(model_type=mt_enum.MOLFORMER, logging=vars(log_config),
                                                           run_type=rm_enum.TRANSFER_LEARNING, version="3.0",
                                                           parameters=vars(self.parameters))
        self.runner = TransferLearningModeConstructor(self.general_config)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _model_saved_and_logs_exist(self):
        self.assertTrue(os.path.isfile(self.parameters.output_model_path))
        self.assertTrue(os.path.isdir(self.log_dir))
        self.assertEqual(count_empty_files(self.log_dir), 0)

    def test_no_validation(self):
        self.parameters.validation_smiles_path = None
        self.runner.run()
        self._model_saved_and_logs_exist()

    def test_with_validation(self):
        self.runner.run()
        self._model_saved_and_logs_exist()
