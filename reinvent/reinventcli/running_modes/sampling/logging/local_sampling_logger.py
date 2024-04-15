from typing import List

import math
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.configurations.logging.sampling_log_configuration import SamplingLoggerConfiguration
from running_modes.sampling.logging.base_sampling_logger import BaseSamplingLogger
from reinvent_chemistry.logging import add_mols, fraction_valid_smiles

from running_modes.utils.plot import plot_histogram, plot_scatter


class LocalSamplingLogger(BaseSamplingLogger):
    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        self._summary_writer = self._instantiate_summary_writer(configuration)

    def log_message(self, message: str):
        self._logger.info(message)

    def timestep_report(self, smiles: List[str], canonical_smiles: List[str], **kwargs):
        self._log_timestep(smiles, canonical_smiles, **kwargs)

    def _log_timestep(self, smiles: List[str], canonical_smiles: List[str], **kwargs):
        valid_smiles_fraction = fraction_valid_smiles(smiles)
        fraction_unique_entries = self._get_unique_entires_fraction(smiles)
        self._visualize_structures(smiles)
        # unique smiles% not necessarily = unique molecule% because it might be possible to generate non-canonical smiles
        self._summary_writer.add_text('Data', f'Valid SMILES: {valid_smiles_fraction}% '
                                              f'Unique SMILES: {fraction_unique_entries}%  '
                                              f'Unique Molecules: {len(set(canonical_smiles))*100/len(smiles)}%  ')
        if 'Time' in kwargs.keys():
            self._summary_writer.add_text('Time', f"{str(kwargs['Time'])}s")
        # Plot histogram
        title = ''
        xlabel = ''
        for key, value in kwargs.items():
            if 'Tanimoto' in key:
                bins = np.arange(0, 11) * 0.1
                xlabel = 'Tanimoto'
            elif 'Output_likelihood' in key:
                bins = range(math.floor(min(value)), math.ceil(max(value)) + 2)
                xlabel = 'Output_likelihood'
            else:
                bins = 50

            if 'valid' in key:
                title = 'valid'
            elif 'unique' in key:
                title = 'unique'

            if key != 'Time':
                figure = plot_histogram(value, xlabel, bins, title=f'{len(value)} {title}')
                self._summary_writer.add_figure(key, figure)
        # Scatter plot
        xlabel, ylabel = 'Tanimoto', 'Output_likelihood'
        x_key, y_key = f'{xlabel}_unique', f'{ylabel}_unique'
        if x_key in kwargs.keys() and y_key in kwargs.keys():
            x, y = kwargs[x_key], kwargs[y_key]
            figure = plot_scatter(x, y, xlabel=xlabel, ylabel=ylabel, title=f'{len(x)} Unique')
            self._summary_writer.add_figure(f'{xlabel}_{ylabel}', figure)

    def _visualize_structures(self, smiles):
        list_of_labels, list_of_mols = self._count_unique_inchi_keys(smiles)
        if len(list_of_mols) > 0:
            add_mols(self._summary_writer, "Most Frequent Molecules", list_of_mols, self._rows, list_of_labels)

    def _instantiate_summary_writer(self, configuration):
        log_config = SamplingLoggerConfiguration(**configuration.logging)
        return SummaryWriter(log_dir=log_config.logging_path)