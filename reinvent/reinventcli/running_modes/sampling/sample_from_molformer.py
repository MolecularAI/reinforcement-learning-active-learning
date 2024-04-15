import os
import time
from typing import List

import pandas as pd
import torch.utils.data as tud

from reinvent_chemistry import Conversions
from reinvent_chemistry.similarity import Similarity
from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.molformer_adapter import MolformerAdapter

from reinvent_models.molformer.dataset.dataset import Dataset
from reinvent_models.molformer.enums import SamplingModesEnum
from reinvent_models.molformer.models.vocabulary import SMILESTokenizer

from running_modes.automated_curriculum_learning.dto import SampledSequencesDTO
from running_modes.configurations.compound_sampling.multinomial_sampling_configuration import \
    MultinomialSamplingConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.compound_sampling.sample_from_model_configuration import SampleFromModelConfiguration
from running_modes.sampling.logging.sampling_logger import SamplingLogger


class SampleFromMolformerRunner(BaseRunningMode):

    def __init__(self, main_config: GeneralConfigurationEnvelope, configuration: SampleFromModelConfiguration):
        self._model = MolformerAdapter(configuration.model_path, mode=ModelModeEnum.INFERENCE)
        self._output_path = configuration.output_smiles_path
        self._num_smiles = configuration.num_smiles
        self._batch_size = configuration.batch_size
        self._with_likelihood = configuration.with_likelihood
        self._sample_strategy = configuration.sampling_strategy
        self._drop_duplicate = configuration.drop_duplicate
        self._logger = SamplingLogger(main_config)
        self._parameters = configuration.parameters
        self._target_smiles_path = self._parameters.get('target_smiles_path', "")

        self._conversion = Conversions()
        self._input = [self._conversion.convert_to_standardized_smiles(smile) for smile in configuration.input]

        config = MultinomialSamplingConfiguration.parse_obj(self._parameters)
        self._model.set_temperature(config.temperature)

        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)

    def _sample(self) -> List[SampledSequencesDTO]:
        if self._sample_strategy == SamplingModesEnum.MULTINOMIAL:
            smiles = self._input * self._num_smiles
        elif self._sample_strategy == SamplingModesEnum.BEAMSEARCH:
            smiles = self._input
            self._model.set_beam_size(self._num_smiles)

        tokenizer = SMILESTokenizer()
        dataset = Dataset(smiles, self._model.get_vocabulary(), tokenizer)
        dataloader = tud.DataLoader(
            dataset,
            batch_size=min(self._batch_size, len(smiles)),
            shuffle=False,
            collate_fn=Dataset.collate_fn,
        )
        sampled_sequences = []
        for batch in dataloader:
            src, src_mask = batch
            if src.device != self._model.device:
                src = src.to(self._model.device)
            if src_mask.device != self._model.device:
                src_mask = src_mask.to(self._model.device)
            sampled_sequences_batch = self._model.sample(src, src_mask, self._sample_strategy)
            sampled_sequences.extend(sampled_sequences_batch)

        return sampled_sequences

    def check_nll(self, target_smiles):
        # Prepare input for checking likelihood_smiles
        dto_list = []
        reference_compounds = self._input
        for compound in reference_compounds:
            for smi in target_smiles:
                current_smi = smi

                try:
                    cano_smi = self._conversion.convert_to_rdkit_smiles(smi, sanitize=True, isomericSmiles=True)
                    current_smi = cano_smi
                # FIXME: catch appropriate exception from RDKit
                except Exception:
                    print(f"WARNING. SMILES {smi} is invalid")

                try:
                    tokenized_smi = self._model.tokenizer.tokenize(current_smi)
                    self._model.vocabulary.encode(tokenized_smi)
                except KeyError as e:
                    print(f"WARNING. SMILES {current_smi} contains an invalid token {e}. It will be ignored")
                else:
                    dto_list.append(SampledSequencesDTO(compound, current_smi, 0))

        df_dict = {
            "Input": [dto.input for dto in dto_list],
            "Target": [dto.output for dto in dto_list]
        }

        # Check NLL of provided target smiles given input, smiles with unknown tokens will be ignored
        i = 0
        results = []
        while i < len(dto_list):
            i_right = min(len(dto_list), i + self._batch_size)
            if i < i_right:
                batch_dto_list = dto_list[i:i_right]
                batch_results = self._model.likelihood_smiles(batch_dto_list)
                results.extend(batch_results.likelihood.cpu().detach().numpy())
                i = min(len(dto_list), i + self._batch_size)
            else:
                break

        # Create dataframe
        # FIXME: avoid using pandas dataframe in Reinvent4
        df_dict['Negative_log_likelihood'] = results
        df = pd.DataFrame(df_dict)

        # Compute Tanimoto
        df['Tanimoto'] = None
        valid_mols, valid_idxs = self._conversion.smiles_to_mols_and_indices(df['Target'].tolist())
        scores = []
        for idx in valid_idxs:
            row = df.iloc[idx]
            scores.extend(self._calculate_tanimoto([row['Input']], [row['Target']]))
        df.loc[valid_idxs, 'Tanimoto'] = scores

        # Save to file
        parent_dir = os.path.dirname(self._output_path)
        df.to_csv(os.path.join(parent_dir, f'target_nll.csv'), index=False)

    def _calculate_tanimoto(self, reference_smiles, smiles):
        # compute Tanimoto similarity between reference_smiles and smiles;
        # return largest if multiple reference smiles provided
        specific_parameters = {"radius": 2, "use_features": False}
        ref_fingerprints = self._conversion.smiles_to_fingerprints(reference_smiles,
                                                                         radius=specific_parameters['radius'],
                                                                         use_features=specific_parameters[
                                                                             'use_features'])
        valid_mols, valid_idxs = self._conversion.smiles_to_mols_and_indices(smiles)
        query_fps = self._conversion.mols_to_fingerprints(valid_mols, radius=specific_parameters['radius'],
                                                            use_features=specific_parameters[
                                                                'use_features'])
        similarity = Similarity()
        scores = similarity.calculate_tanimoto(query_fps, ref_fingerprints)
        return scores

    def run(self):
        start_time = time.time()
        sampled_sequences = self._sample()
        input_smiles = [dto.input for dto in sampled_sequences]
        output_smiles = [dto.output for dto in sampled_sequences]
        likelihoods = [dto.nll for dto in sampled_sequences]

        # canonical_smiles
        canonical_output_smiles = []
        for smi in output_smiles:
            try:
                cano_smi = self._conversion.convert_to_rdkit_smiles(smi, sanitize=True, isomericSmiles=True)
                canonical_output_smiles.append(cano_smi)
            # FIXME: catch appropriate exception from RDKit in Reinvent4
            except Exception:
                canonical_output_smiles.append(None)

        # compute Tanimoto similarity between generated compounds and input compounds; return largest
        valid_mols, valid_idxs = self._conversion.smiles_to_mols_and_indices(output_smiles)
        scores = self._calculate_tanimoto(self._input, output_smiles)

        # build dataframe
        # FIXME: avoid using pandas dataframe in Reinvent4
        results = list(zip(input_smiles, output_smiles, likelihoods, canonical_output_smiles)) \
            if self._with_likelihood else list(zip(input_smiles, output_smiles, canonical_output_smiles))
        columns = ['Input', 'Output', 'Output_likelihood', 'Canonical_output'] \
            if self._with_likelihood else ['Input', 'Output', 'Canonical_output']
        df = pd.DataFrame(results, columns=columns)
        df['Tanimoto'] = None
        df.loc[valid_idxs, 'Tanimoto'] = scores

        # Valid
        df.dropna(subset=['Canonical_output'], inplace=True)

        # drop duplicated
        df_unique = df.drop_duplicates(subset=['Canonical_output'])

        # Write to file
        if self._drop_duplicate:
            df_unique.to_csv(self._output_path, index=False)
        else:
            df.to_csv(self._output_path, index=False)

        # Log all valid molecules' stat, include duplicated
        kwargs = {'Tanimoto_valid': df['Tanimoto'].tolist()}
        # Log all unique molecules' stat
        kwargs['Tanimoto_unique'] = df_unique['Tanimoto'].tolist()
        if self._with_likelihood:
            kwargs['Output_likelihood_valid'] = df['Output_likelihood'].tolist()
            kwargs['Output_likelihood_unique'] = df_unique['Output_likelihood'].tolist()
        kwargs['Time'] = int(time.time()-start_time)
        self._logger.timestep_report(output_smiles,
                                     df['Canonical_output'].tolist(),
                                     **kwargs
                                     )
        self._logger.log_out_input_configuration()

        # check NLL for target smiles if provided
        target_smiles = []
        if self._target_smiles_path:
            with open(self._target_smiles_path) as f:
                target_smiles = [line.rstrip('\n') for line in f]
        if len(target_smiles) > 0:
            self.check_nll(target_smiles)
