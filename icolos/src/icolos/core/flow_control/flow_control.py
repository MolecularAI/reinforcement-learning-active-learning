from typing import Any, Optional
from pydantic import BaseModel, PrivateAttr
from icolos.core.workflow_steps.step import StepSettingsParameters
from icolos.core.workflow_steps.step import StepBase
from icolos.loggers.steplogger import StepLogger
from icolos.core.workflow_steps.step import (
    StepData,
    StepInputParameters,
    StepWriteoutParameters,
    StepExecutionParameters,
)


class BaseStepConfig(BaseModel):
    """
    Minimal template class for the base config, without the unnecessary stuff that StepBase requires
    """

    step_id: Optional[str] = None
    work_dir: Optional[str] = None
    type: Optional[str] = None
    data: StepData = StepData()
    input: StepInputParameters = StepInputParameters()
    writeout: list[StepWriteoutParameters] = []
    execution: StepExecutionParameters = StepExecutionParameters()
    settings: StepSettingsParameters = StepSettingsParameters()

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "type": self.type,
            "execution": self.execution,
            "settings": self.settings,
            "work_dir": self.work_dir,
            "data": self.data,
            "input": self.input,
            "writeout": self.writeout,
        }


class FlowControlBase(BaseModel):
    # List of steps to be iterated over, each set needs their inputs chained together
    base_config: list[BaseStepConfig] = None
    initialized_steps: list[StepBase] = None
    _logger = PrivateAttr()

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._logger = StepLogger()
