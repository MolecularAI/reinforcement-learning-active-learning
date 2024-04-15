import time
from typing import List, Union, Tuple
import numpy as np

import torch
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO
from reinvent_scoring import FinalSummary
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.loggable_data_dto import UpdateLoggableDataDTO
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.actions.molformer_sample_model import MolformerSampleModel
from running_modes.automated_curriculum_learning.dto import TimestepDTO, UpdatedLikelihoodsDTO
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy import LearningStrategy
from running_modes.automated_curriculum_learning.production_strategy.base_production_strategy import \
    BaseProductionStrategy
from running_modes.reinforcement_learning.learning_strategy import BaseLearningStrategy


class MolformerProductionStrategy(BaseProductionStrategy):
    def __init__(self, prior, diversity_filter, inception, scoring_function, configuration, logger):
        super().__init__(prior, diversity_filter, inception, scoring_function, configuration, logger)
        self._parameters.input = \
            [self._conversion.convert_to_standardized_smiles(smile) for smile in self._parameters.input]

    def run(self, cl_agent: GenerativeModelBase, steps_so_far: int):
        self.disable_prior_gradients()
        step_limit = steps_so_far + self._parameters.number_of_steps
        optimizer = torch.optim.Adam(cl_agent.get_network_parameters(), lr=self._parameters.learning_rate)
        learning_strategy = LearningStrategy(self._prior, optimizer, self._parameters.learning_strategy, self._logger)

        for step in range(steps_so_far, step_limit):
            start_time = time.time()
            self.take_step(agent=cl_agent, scoring_function=self._scoring_function, step=step, start_time=start_time,
                           learning_strategy=learning_strategy)

        self._logger.log_message(f"Production finished at step {step_limit}")
        self._logger.save_final_state(cl_agent, self._diversity_filter)

    def take_step(self, agent: GenerativeModelBase, scoring_function: BaseScoringFunction,
                  step: int, start_time: float, learning_strategy) -> float:
        # 1. Sampling
        sampled_sequences = self._sampling(agent)
        # 2. Scoring
        score_summary = self._scoring(scoring_function, sampled_sequences, step, learning_strategy)
        # 3. Updating
        dto = self._updating(sampled_sequences, score_summary.total_score, learning_strategy, agent)
        # 4. Logging
        self._logging(start_time, step, score_summary, dto, agent)

        score = score_summary.total_score.mean()
        return score

    def _sampling(self, agent) -> List[SampledSequencesDTO]:
        sampling_action = MolformerSampleModel(agent, self._parameters.batch_size, self._logger,
                                               self._parameters.randomize_input, self._parameters.sample_strategy)
        sampled_sequences = sampling_action.run(self._parameters.input)
        return sampled_sequences

    def _scoring(self, scoring_function, sampled_sequences: List[SampledSequencesDTO], step: int, learning_strategy) \
            -> FinalSummary:
        smiles = [dto.output for dto in sampled_sequences]
        score_summary = scoring_function.get_final_score_for_step(smiles, step)
        distance_penalty, prior_nll = self._get_distance_to_prior(learning_strategy, sampled_sequences,
                                                       self._parameters.distance_threshold)
        loggable_data = self._prepare_loggable_data(sampled_sequences, prior_nll)
        dto = UpdateDiversityFilterDTO(score_summary, loggable_data, step)
        score_summary.total_score = self._diversity_filter.update_score(dto)
        score_summary.total_score = score_summary.total_score * distance_penalty
        return score_summary

    def _updating(self, sampled_sequences, score, learning_strategy, agent) -> UpdatedLikelihoodsDTO:
        likelihood_dto = agent.likelihood_smiles(sampled_sequences)
        dto = learning_strategy.run(likelihood_dto, score)
        return dto

    def _logging(self, start_time, step, score_summary, likelihood_dto: UpdatedLikelihoodsDTO, agent):
        report_dto = TimestepDTO(start_time, self._parameters.number_of_steps, step, score_summary,
                                 likelihood_dto.agent_likelihood, likelihood_dto.prior_likelihood,
                                 likelihood_dto.augmented_likelihood)
        self._logger.timestep_report(report_dto, self._diversity_filter, agent)

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