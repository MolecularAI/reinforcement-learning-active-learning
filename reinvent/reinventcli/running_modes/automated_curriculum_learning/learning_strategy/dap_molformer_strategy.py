import torch
from reinvent_models.model_factory.dto import BatchLikelihoodDTO

from running_modes.automated_curriculum_learning.dto import UpdatedLikelihoodsDTO
from running_modes.automated_curriculum_learning.learning_strategy.base_double_query_learning_strategy import \
    BaseDoubleQueryLearningStrategy
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration


class DAPMolformerStrategy(BaseDoubleQueryLearningStrategy):

    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger=None):
        """
        TODO: Provide description of the current strategy
        """
        super().__init__(critic_model, optimizer, configuration, logger)

        self._sigma = self._configuration.parameters.get("sigma", 120)

    def _calculate_loss(self, likelihood_dto: BatchLikelihoodDTO, score) -> UpdatedLikelihoodsDTO:
        batch = likelihood_dto.batch
        critic_nlls = self.critic_model.likelihood(batch.input, batch.input_mask, batch.output, batch.output_mask)
        negative_critic_nlls = -critic_nlls
        negative_actor_nlls = -likelihood_dto.likelihood
        augmented_nlls = negative_critic_nlls + self._sigma * self._to_tensor(score)
        loss = torch.pow((augmented_nlls - negative_actor_nlls), 2)
        loss = loss.mean()
        dto = UpdatedLikelihoodsDTO(negative_actor_nlls, negative_critic_nlls, augmented_nlls, loss)
        return dto