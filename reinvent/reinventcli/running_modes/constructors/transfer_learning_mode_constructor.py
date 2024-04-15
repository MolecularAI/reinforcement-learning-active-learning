import torch
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_models.model_factory.generative_model import GenerativeModel
from reinvent_models.reinvent_core.models.model import Model

from running_modes.configurations.transfer_learning.link_invent_transfer_learning_configuration import \
    LinkInventTransferLearningConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.configurations import GeneralConfigurationEnvelope, TransferLearningConfiguration
from running_modes.transfer_learning.link_invent_transfer_learning_runner import LinkInventTransferLearningRunner
from running_modes.transfer_learning.logging.transfer_learning_logger import TransferLearningLogger
from running_modes.transfer_learning.transfer_learning_runner import TransferLearningRunner

# Molformer transfer learning
from running_modes.configurations.transfer_learning.molformer_transfer_learning_configuration import \
    MolformerTransferLearningConfiguration
from running_modes.transfer_learning.molformer_transfer_learning_runner import MolformerTransferLearningRunner
from running_modes.transfer_learning.optimizers import NoamOpt

from dacite import from_dict
from running_modes.utils import set_default_device_cuda


class TransferLearningModeConstructor:
    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        cls._configuration = configuration
        set_default_device_cuda()
        model_type = ModelTypeEnum()
        model_mode = ModelModeEnum()

        logger = TransferLearningLogger(cls._configuration)
        set_default_device_cuda()

        if cls._configuration.model_type == model_type.DEFAULT:
            config = from_dict(data_class=TransferLearningConfiguration, data=cls._configuration.parameters)
            model = Model.load_from_file(config.input_model_path)
            runner = TransferLearningRunner(model, config, logger)

        elif cls._configuration.model_type == model_type.LIB_INVENT:
            raise NotImplementedError(f"Running mode not implemented for a model type: {cls._configuration.model_type}")

        elif cls._configuration.model_type == model_type.LINK_INVENT:
            config = from_dict(data_class=LinkInventTransferLearningConfiguration, data=cls._configuration.parameters)
            model_config = ModelConfiguration(model_type=cls._configuration.model_type, model_mode=model_mode.TRAINING,
                                              model_file_path=config.empty_model)
            model = GenerativeModel(model_config)
            optimizer = torch.optim.Adam(model.get_network_parameters(), lr=config.learning_rate.start)
            learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate.step,
                                                                       gamma=config.learning_rate.gamma)
            runner = LinkInventTransferLearningRunner(configuration=config, model=model, logger=logger,
                                                      optimizer=optimizer,
                                                      learning_rate_scheduler=learning_rate_scheduler)
        elif cls._configuration.model_type == model_type.MOLFORMER:
            config = from_dict(data_class=MolformerTransferLearningConfiguration, data=cls._configuration.parameters)
            model_config = ModelConfiguration(model_type=cls._configuration.model_type, model_mode=model_mode.TRAINING,
                                              model_file_path=config.input_model_path)

            model = GenerativeModel(model_config)
            opt_config = config.optimizer
            d_model = model.network.encoder.layers[0].self_attn.linears[0].in_features
            adam = torch.optim.Adam(model.get_network_parameters(), lr=opt_config.lr,
                                    betas=(opt_config.beta1, opt_config.beta2), eps=opt_config.eps)
            # If optimizer_chepoint is passed then use it here
            optimizer = NoamOpt(d_model, opt_config.factor, opt_config.warmup, adam)
            if not config.reset_optimizer:
                opt_filename = config.input_model_path + '.opt'
                if os.path.exists(opt_filename):
                    # The load_state_dict function effects the
                    # internal state of Adam, the rate, and the step
                    optimizer.load_state_dict(torch.load(opt_filename))
            runner = MolformerTransferLearningRunner(configuration=config, model=model, logger=logger,
                                                      optimizer=optimizer)
        else:
            raise ValueError(f"Invalid model_type provided: '{cls._configuration.model_type}")

        return runner
