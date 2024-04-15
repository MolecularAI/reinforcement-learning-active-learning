from dataclasses import dataclass
from running_modes.configurations.transfer_learning.noamopt_configuration import NoamoptConfiguration

#FIXME: pairs must be a configuration object not a generic dict 

@dataclass
class MolformerTransferLearningConfiguration:
    input_model_path: str
    output_model_path: str
    input_smiles_path: str
    optimizer: NoamoptConfiguration
    pairs: dict
    reset_optimizer: bool = True
    ranking_loss_penalty: bool = False
    starting_epoch: int = 1
    num_epochs: int = 60
    batch_size: int = 128
    shuffle_each_epoch: bool = True
    save_every_n_epochs: int = 0
    validation_percentage: float = 0.0
    validation_seed: int = None
