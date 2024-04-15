import os
import pandas as pd

from reinvent_chemistry.file_reader import FileReader
from reinvent_models.molformer.molformer_model import EncoderDecoder, MolformerModel
from reinvent_models.molformer.models.vocabulary import Vocabulary, SMILESTokenizer

from running_modes.configurations.create_model.molformer_create_model_configuration import \
    MolformerCreateModelConfiguration
from running_modes.create_model.logging.base_create_model_logger import BaseCreateModelLogger
from reinvent_models.model_factory.enums.model_parameter_enum import ModelParametersEnum


class MolformerCreateModelRunner:
    def __init__(self, configuration: MolformerCreateModelConfiguration, logger: BaseCreateModelLogger):
        self._configuration = configuration
        self._logger = logger
        self._reader = FileReader([], None)

    def run(self):
        vocabulary = self._build_vocabulary()
        model = self._get_model(vocabulary)
        self._save_model(model)
        return model

    def _build_vocabulary(self) -> Vocabulary:
        self._logger.log_message('Building vocabulary')
        file_name = self._configuration.input_smiles_path
        pd_data = pd.read_csv(file_name, sep=",")
        self._logger.log_message(f"Read {file_name} file")
        # FIXME: we assume that the first column contain
        # the smiles. Does it sound reasonable? 
        smiles_list = pd.unique(pd_data[pd_data.columns[0]].values.ravel('K'))
        self._logger.log_message(f"Number of SMILES in chemical transformations: {len(smiles_list):d}")

        tokenizer = SMILESTokenizer()
        tokens = set()
        for smi in smiles_list:
            tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))
        vocabulary = Vocabulary()
        vocabulary.update(["*", "^", "$"] + sorted(tokens))  # pad=0, start=1, end=2
        # for random smiles
        if "8" not in vocabulary.tokens():
            vocabulary.update(["8"])
        return vocabulary

    def _get_model(self, model_vocabulary: Vocabulary) -> MolformerModel:
        self._configuration.network.vocabulary_size = len(model_vocabulary)
        network = EncoderDecoder(**vars(self._configuration.network))
        model = MolformerModel(vocabulary=model_vocabulary, network=network,
                               max_sequence_length=self._configuration.max_sequence_length,
                               no_cuda=not self._configuration.use_cuda)
        return model

    def _save_model(self, model: MolformerModel):
        self._logger.log_message(f'Saving model at {self._configuration.output_model_path}')
        os.makedirs(os.path.dirname(self._configuration.output_model_path), exist_ok=True)
        model.save_to_file(self._configuration.output_model_path)
        self._logger.log_out_input_configuration()
