from dataclasses import dataclass

from reinvent_models.molformer.dto.molformer_model_parameters_dto import MolformerNetworkParameters

@dataclass
class MolformerCreateModelConfiguration:
    input_smiles_path: str
    output_model_path: str
    network: MolformerNetworkParameters
    max_sequence_length: int = 128
    use_cuda: bool = True
