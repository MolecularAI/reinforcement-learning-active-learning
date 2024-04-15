from dacite import from_dict

from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.configurations import GeneralConfigurationEnvelope, SampleFromModelConfiguration
from running_modes.enums.model_type_enum import ModelTypeEnum
from running_modes.sampling.sample_from_model import SampleFromModelRunner
from running_modes.sampling.sample_from_molformer import SampleFromMolformerRunner
from running_modes.utils.general import set_default_device_cuda


class SamplingModeConstructor:
    def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        self._configuration = configuration
        config = from_dict(data_class=SampleFromModelConfiguration, data=self._configuration.parameters)

        model_type = ModelTypeEnum()
        if configuration.model_type == model_type.DEFAULT:
            set_default_device_cuda()
            runner = SampleFromModelRunner(self._configuration, config)
        elif configuration.model_type == model_type.MOLFORMER:
            runner = SampleFromMolformerRunner(self._configuration, config)
        return runner