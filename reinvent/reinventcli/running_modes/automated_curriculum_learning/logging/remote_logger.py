import os
import numpy as np
import requests

from reinvent_chemistry.logging import padding_with_invalid_smiles, \
    check_for_invalid_mols_and_create_legend, fraction_valid_smiles, \
    sort_smiles_by_score
from reinvent_scoring.scoring.diversity_filters.lib_invent.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum, ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.score_summary import FinalSummary
from running_modes.automated_curriculum_learning.dto.timestep_dto import TimestepDTO
from running_modes.automated_curriculum_learning.logging.base_logger import BaseLogger
from running_modes.configurations import GeneralConfigurationEnvelope, CurriculumLoggerConfiguration, \
    get_remote_logging_auth_token
from running_modes.utils import _is_development_environment
from running_modes.utils.general import estimate_run_time


class RemoteLogger(BaseLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope, log_config: CurriculumLoggerConfiguration):
        super().__init__(configuration, log_config)
        self._rows = 2
        self._columns = 5
        self._sample_size = self._rows * self._columns
        self._sf_component_enum = ScoringFunctionComponentNameEnum()
        self._specific_parameters_enum = ComponentSpecificParametersEnum()
        self._is_dev = _is_development_environment()

    def log_message(self, message: str):
        self._logger.info(message)

    def timestep_report(self, report_dto: TimestepDTO, diversity_filter: BaseDiversityFilter, agent):
        smiles = report_dto.score_summary.scored_smiles

        score_components = self._score_summary_breakdown(report_dto)
        learning_curves = self._learning_curve_profile(report_dto)
        smiles_report = self._create_sample_report(report_dto)

        time_estimation = estimate_run_time(report_dto.start_time, report_dto.n_steps, report_dto.step)
        data = self._assemble_timestep_report(report_dto.step, score_components, diversity_filter, learning_curves,
                                              time_estimation, fraction_valid_smiles(smiles), smiles_report)
        self._notify_server(data, self._log_config.recipient)
        self.save_checkpoint(report_dto.step, diversity_filter, agent)

    def save_final_state(self, agent, diversity_filter):
        agent.save_to_file(os.path.join(self._log_config.result_folder, 'Agent.ckpt'))
        self.save_filter_memory(diversity_filter)
        self.log_out_input_configuration()

    def _notify_server(self, data, to_address):
        """This is called every time we are posting data to server"""
        try:
            self._logger.warning(f"posting to {to_address}")
            headers = {
                'Accept': 'application/json', 'Content-Type': 'application/json',
                'Authorization': get_remote_logging_auth_token()
            }
            response = requests.post(to_address, json=data, headers=headers)

            if self._is_dev:
                """logs out the response content only when running a test instance"""
                if response.status_code == requests.codes.ok:
                    self._logger.info(f"SUCCESS: {response.status_code}")
                    self._logger.info(response.content)
                else:
                    self._logger.info(f"PROBLEM: {response.status_code}")
                    self._logger.exception(data, exc_info=False)
        except Exception as t_ex:
            self._logger.exception("Exception occurred", exc_info=True)
            self._logger.exception(f"Attempted posting the following data:")
            self._logger.exception(data, exc_info=False)

    def _get_matching_substructure_from_config(self, score_summary: FinalSummary):
        smarts_pattern = ""
        for summary_component in score_summary.scaffold_log:
            if summary_component.parameters.component_type == self._sf_component_enum.MATCHING_SUBSTRUCTURE:
                smarts = summary_component.parameters.specific_parameters.get(self._specific_parameters_enum.SMILES, [])
                if len(smarts) > 0:
                    smarts_pattern = smarts[0]
        return smarts_pattern

    def _create_sample_report(self, dto: TimestepDTO):
        score = dto.score_summary.total_score
        smiles = dto.score_summary.scored_smiles
        score, smiles = sort_smiles_by_score(score, smiles)
        smiles = padding_with_invalid_smiles(smiles, self._sample_size)
        _, legend = check_for_invalid_mols_and_create_legend(smiles, score, self._sample_size)
        smarts_pattern = self._get_matching_substructure_from_config(dto.score_summary)

        smiles_legend_pairs = [{"smiles": smiles[indx], "legend": legend[indx]} for indx in range(self._sample_size)]

        report = {
            "smarts_pattern": smarts_pattern,
            "smiles_legend_pairs": smiles_legend_pairs
        }
        return report

    def _learning_curve_profile(self, dto: TimestepDTO):
        learning_curves = {
            "prior": float(np.float(dto.prior_likelihood.detach().mean().cpu())),
            "augmented": float(np.float(dto.augmented_likelihood.detach().mean().cpu())),
            "agent": float(np.float(dto.agent_likelihood.detach().mean().cpu()))
        }
        return learning_curves

    def _score_summary_breakdown(self, dto: TimestepDTO):
        mean_score = np.mean(dto.score_summary.total_score)
        score_components = {}
        for i, log in enumerate(dto.score_summary.profile):
            score_components[f"{dto.score_summary.profile[i].component_type}:{dto.score_summary.profile[i].name}"] = \
                float(np.mean(dto.score_summary.profile[i].score))
        score_components["total_score:total_score"] = float(mean_score)
        return score_components

    def _assemble_timestep_report(self, step, score_components, diversity_filter:BaseDiversityFilter,
                                  learning_curves, time_estimation, fraction_valid_smiles, smiles_report):
        actual_step = step + 1
        timestep_report = {"step": actual_step,
                           "components": score_components,
                           "learning": learning_curves,
                           "time_estimation": time_estimation,
                           "fraction_valid_smiles": fraction_valid_smiles,
                           "smiles_report": smiles_report,
                           "collected smiles in memory": diversity_filter.number_of_smiles_in_memory()
                           }
        return timestep_report
