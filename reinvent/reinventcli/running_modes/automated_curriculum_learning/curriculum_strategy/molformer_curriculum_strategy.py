import time
from typing import List, Tuple
import numpy as np

import torch
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import FinalSummary
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.loggable_data_dto import UpdateLoggableDataDTO
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.actions.molformer_sample_model import MolformerSampleModel
from running_modes.automated_curriculum_learning.curriculum_strategy.base_curriculum_strategy import \
    BaseCurriculumStrategy
from running_modes.automated_curriculum_learning.dto import CurriculumOutcomeDTO, TimestepDTO, SampledSequencesDTO, \
    UpdatedLikelihoodsDTO
from running_modes.automated_curriculum_learning.learning_strategy.base_learning_strategy import BaseLearningStrategy


class MolformerCurriculumStrategy(BaseCurriculumStrategy):
    def __init__(self, prior, agent, configuration, diversity_filter, inception, logger):
        super().__init__(prior, agent, configuration, diversity_filter, inception, logger)
        self._parameters.input = \
            [self._conversion.convert_to_standardized_smiles(smile) for smile in self._parameters.input]

    def run(self) -> CurriculumOutcomeDTO:
        step_counter = 0
        self.disable_prior_gradients()

        for item_id, sf_configuration in enumerate(self._parameters.curriculum_objectives):
            start_time = time.time()
            scoring_function = self._setup_scoring_function(item_id)
            step_counter = self.promote_agent(agent=self._agent, scoring_function=scoring_function,
                                              step_counter=step_counter, start_time=start_time,
                                              merging_threshold=sf_configuration.score_threshold)
            self.save_and_flush_memory(agent=self._agent, memory_name=f"_merge_{item_id}")
        is_successful_curriculum = step_counter < self._parameters.max_num_iterations
        outcome_dto = CurriculumOutcomeDTO(self._agent, step_counter, successful_curriculum=is_successful_curriculum)

        return outcome_dto

    def take_step(self, agent: GenerativeModelBase, scoring_function: BaseScoringFunction,
                  step:int, start_time: float) -> float:
        # 1. Sampling
        sampled_sequences = self._sampling(agent)
        # 2. Scoring
        score_summary = self._scoring(scoring_function, sampled_sequences, step)
        # 3. Updating
        dto = self._updating(sampled_sequences, score_summary.total_score)
        # 4. Logging
        self._logging(start_time, step, score_summary, dto)

        score = score_summary.total_score.mean()
        return score

    def _sampling(self, agent) -> List[SampledSequencesDTO]:
        sampling_action = MolformerSampleModel(agent, self._parameters.batch_size, self._logger,
                                               self._parameters.randomize_input, self._parameters.sample_strategy)
        sampled_sequences = sampling_action.run(self._parameters.input)
        return sampled_sequences

    def _scoring(self, scoring_function, sampled_sequences: List[SampledSequencesDTO], step: int) -> FinalSummary:
        smiles = [dto.output for dto in sampled_sequences]
        score_summary = scoring_function.get_final_score_for_step(smiles, step)
        distance_penalty, prior_nll = self._get_distance_to_prior(self.learning_strategy, sampled_sequences,
                                                       self._parameters.distance_threshold)
        loggable_data = self._prepare_loggable_data(sampled_sequences, prior_nll)

        dto = UpdateDiversityFilterDTO(score_summary, loggable_data, step)
        score_summary.total_score = self._diversity_filter.update_score(dto)
        score_summary.total_score = score_summary.total_score * distance_penalty
        return score_summary

    def _updating(self, sampled_sequences, score) -> UpdatedLikelihoodsDTO:
        likelihood_dto = self._agent.likelihood_smiles(sampled_sequences)
        dto = self.learning_strategy.run(likelihood_dto, score)
        return dto

    def _logging(self, start_time, step, score_summary, likelihood_dto: UpdatedLikelihoodsDTO):
        report_dto = TimestepDTO(start_time, self._parameters.max_num_iterations, step, score_summary,
                                 likelihood_dto.agent_likelihood, likelihood_dto.prior_likelihood,
                                 likelihood_dto.augmented_likelihood)
        self._logger.timestep_report(report_dto, self._diversity_filter, self._agent)

    def _get_distance_to_prior(self, learning_strategy: BaseLearningStrategy, sampled_dto: List[SampledSequencesDTO],
                               distance_threshold: float) -> Tuple[np.ndarray, torch.Tensor]:
        dto = learning_strategy.critic_model.likelihood_smiles(sampled_dto)

        if isinstance(dto.likelihood, torch.Tensor):
            ones = torch.ones_like(dto.likelihood, requires_grad=False)
            mask = torch.where(dto.likelihood < distance_threshold, ones, distance_threshold / dto.likelihood)
            mask = mask.cpu().numpy()
        else:
            ones = np.ones_like(dto.likelihood)
            mask = np.where(dto.likelihood < distance_threshold, ones, distance_threshold / dto.likelihood)
        return mask, dto.likelihood

    def _prepare_loggable_data(self, sampled_sequences: List[SampledSequencesDTO], prior_nll: torch.Tensor) -> \
            List[UpdateLoggableDataDTO]:
        loggable_data = [UpdateLoggableDataDTO(dto.input, dto.output, dto.nll, prior_nll[i]) for i, dto in enumerate(sampled_sequences)]
        return loggable_data