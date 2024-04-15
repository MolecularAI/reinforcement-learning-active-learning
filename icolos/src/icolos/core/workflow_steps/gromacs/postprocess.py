from pathlib import Path
from pydantic import BaseModel

from icolos.utils.enums.step_enums import StepGromacsEnum
from icolos.utils.execute_external.execute import Executor
from icolos.core.workflow_steps.step import _LE
from icolos.core.workflow_steps.gromacs.base import StepGromacsBase

_SGE = StepGromacsEnum()


class StepGMXPostProcess(StepGromacsBase, BaseModel):
    """
    Run a trajectory analysis script of the users choice,
    based on xtc and gro file input.

    Input
    -----
    :gmx_state source: Step ID of the previous Gromacs step

    Settings
    --------
    :additional script_path: Path to the script to run
    :additional trajectory_flag:
        Command argument referring to the trajectory, default = "-f"
    :additional structure_flag:
        Command argument referring to the structure, default = "-s"

    """

    def __init__(self, **data):
        super().__init__(**data)

        self._initialize_backend(executor=Executor)
        self._check_backend_availability()

    def execute(self):
        tmp_dir = self._make_tmpdir()
        topol = self.get_topol()

        # We assume the script takes a trajectory (in xtc format)
        # and a structure (in gro format) as input
        topol.write_structure(tmp_dir)
        topol.write_trajectory(tmp_dir)
        self.write_input_files(tmp_dir)

        # Because we don't know how these are passed to the script,
        # we ask the user to specify the exact flags
        trajectory_flag = self.settings.additional.get("trajectory_flag", "-f")
        structure_flag = self.settings.additional.get("structure_flag", "-s")
        script_path = Path(self.settings.additional.get("script_path"))
        if not script_path.exists():
            raise ValueError(f"Can't find script under {script_path.as_posix()}")

        # Any number of additional flags and arguments can be passed to the script
        arguments = [trajectory_flag, _SGE.STD_XTC, structure_flag, _SGE.STD_STRUCTURE]
        for key, value in self.settings.arguments.parameters.items():
            arguments.extend([key, value])
        if self.settings.arguments.flags:
            arguments.extend(self.settings.arguments.flags)

        result = self._backend_executor.execute(
            command=script_path.as_posix(),
            arguments=arguments,
            location=tmp_dir
        )

        for line in result.stdout.split("\n"):
            self._logger_blank.log(line, _LE.DEBUG)
        self._logger.log(
            f"Completed execution for {self.step_id} successfully", _LE.INFO
        )

        self._parse_output(tmp_dir)
        self._remove_temporary(tmp_dir)

