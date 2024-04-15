import os
import shutil
import unittest
import pytest
import pandas as pd

from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.sampling_log_configuration import SamplingLoggerConfiguration
from running_modes.configurations.compound_sampling.sample_from_model_configuration import SampleFromModelConfiguration
from running_modes.sampling.sample_from_molformer import SampleFromMolformerRunner
from unittest_reinvent.fixtures.test_config import MAIN_TEST_PATH, MOLFORMER_PRIOR_PATH
from running_modes.enums.logging_mode_enum import LoggingModeEnum
from running_modes.enums.running_mode_enum import RunningModeEnum
from reinvent_models.molformer.enums import SamplingModesEnum
from unittest_reinvent.fixtures.test_data import CELECOXIB


@pytest.mark.integration
class TestSampleFromModel(unittest.TestCase):

    def setUp(self):
        rm_enums = RunningModeEnum()
        lm_enums = LoggingModeEnum()
        self.workfolder = MAIN_TEST_PATH
        self.output_file = os.path.join(self.workfolder, "sample.csv")
        self.num_smiles = 100
        self.config = SampleFromModelConfiguration(model_path=MOLFORMER_PRIOR_PATH,
                                                   output_smiles_path=self.output_file, num_smiles=self.num_smiles,
                                                   batch_size=64,
                                                   with_likelihood=False,
                                                   sampling_strategy=SamplingModesEnum.MULTINOMIAL,
                                                   input=[CELECOXIB]
                                                   )
        self.logging = SamplingLoggerConfiguration(recipient=lm_enums.LOCAL,
                                                   logging_path=f"{self.workfolder}/log", job_name="test_job")
        self.configurationenvelope = GeneralConfigurationEnvelope(parameters=vars(self.config), logging=vars(self.logging),
                                                                  run_type=rm_enums.SAMPLING, version="2.0")

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def _check_file_and_count_lines(self):
        self.assertTrue(os.path.isfile(self.output_file))
        num_lines = sum(1 for _ in open(self.output_file))
        self.assertLessEqual(num_lines, self.num_smiles+1)


    def _check_columns(self, columns):
        df = pd.read_csv(self.output_file)
        self.assertListEqual(list(df.columns), columns)


    def test_sample_from_model_multinomial_without_likelihood(self):
        runner = SampleFromMolformerRunner(self.configurationenvelope, self.config)
        runner.run()

        self._check_file_and_count_lines()
        self._check_columns(['Input', 'Output', 'Canonical_output', 'Tanimoto'])


    def test_sample_from_model_multinomial_with_likelihood(self):
        self.config.with_likelihood = True
        runner = SampleFromMolformerRunner(self.configurationenvelope, self.config)
        runner.run()

        self._check_file_and_count_lines()
        self._check_columns(['Input', 'Output', 'Output_likelihood', 'Canonical_output', 'Tanimoto'])


    def test_sample_from_model_multinomial_temperature_with_likelihood(self):
        self.config.with_likelihood = True
        self.config.parameters = {'temperature': 1.5}
        runner = SampleFromMolformerRunner(self.configurationenvelope, self.config)
        runner.run()

        self._check_file_and_count_lines()
        self._check_columns(['Input', 'Output', 'Output_likelihood', 'Canonical_output', 'Tanimoto'])


    def test_sample_from_model_beamsearch_with_likelihood(self):
        self.config.with_likelihood = True
        self.config.sampling_strategy = SamplingModesEnum.BEAMSEARCH
        runner = SampleFromMolformerRunner(self.configurationenvelope, self.config)
        runner.run()

        self._check_file_and_count_lines()
        self._check_columns(['Input', 'Output', 'Output_likelihood', 'Canonical_output', 'Tanimoto'])
