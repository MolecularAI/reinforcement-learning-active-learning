from typing import List

import numpy as np
import torch.utils.data as tud
from reinvent_chemistry import Conversions
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.molformer.dataset.dataset import Dataset
from reinvent_models.molformer.enums import SamplingModesEnum
from reinvent_models.molformer.models.vocabulary import SMILESTokenizer
from running_modes.automated_curriculum_learning.dto import SampledSequencesDTO
from running_modes.automated_curriculum_learning.actions import BaseSampleAction


class MolformerSampleModel(BaseSampleAction):
    def __init__(
        self,
        model: GenerativeModelBase,
        batch_size: int,
        logger=None,
        randomize=False,
        sample_strategy=SamplingModesEnum.MULTINOMIAL,
        sample_uniquely=True,
    ):
        """
        Creates an instance of SampleModel.
        :params model: A model instance.
        :params batch_size: Batch size to use.
        :return:
        """
        super().__init__(logger)
        self.model = model
        self._batch_size = batch_size
        self._randomize = randomize
        self._sample_strategy = sample_strategy
        self._sample_uniquely = sample_uniquely
        self._conversions = Conversions()

    def run(self, smiles: List[str]) -> List[SampledSequencesDTO]:
        smiles = (
            [self._randomize_smile(smile) for smile in smiles]
            if self._randomize
            else smiles
        )

        if self._sample_strategy == SamplingModesEnum.MULTINOMIAL:
            smiles = smiles * self._batch_size
        elif self._sample_strategy == SamplingModesEnum.BEAMSEARCH:
            self.model.set_beam_size(self._batch_size)
        else:
            raise ValueError(
                f"Sample strategy `{self._sample_strategy}` is not implemented in reinventcli"
            )

        tokenizer = SMILESTokenizer()
        dataset = Dataset(smiles, self.model.get_vocabulary(), tokenizer)
        dataloader = tud.DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            collate_fn=Dataset.collate_fn,
        )
        sampled_sequences = []

        for batch in dataloader:
            src, src_mask = batch
            if src.device != self.model.device:
                src = src.to(self.model.device)
            if src_mask.device != self.model.device:
                src_mask = src_mask.to(self.model.device)
            sampled_sequences = self.model.sample(src, src_mask, self._sample_strategy)

        unique_sequences = self._sample_unique_sequences(sampled_sequences)

        return unique_sequences

    def _sample_unique_sequences(
        self, sampled_sequences: List[SampledSequencesDTO]
    ) -> List[SampledSequencesDTO]:
        smiles = [dto.output for dto in sampled_sequences]
        unique_idxs = self._get_indices_of_unique_smiles(smiles)
        sampled_sequences_np = np.array(sampled_sequences)
        unique_sampled_sequences = sampled_sequences_np[unique_idxs]
        return unique_sampled_sequences.tolist()

    def _randomize_smile(self, smile: str):
        input_mol = self._conversions.smile_to_mol(smile)
        randomized_smile = self._conversions.mol_to_random_smiles(input_mol)
        return randomized_smile
