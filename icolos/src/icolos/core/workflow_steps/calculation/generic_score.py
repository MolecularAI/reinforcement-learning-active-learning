from pathlib import Path
import tempfile
from pydantic import BaseModel
from rdkit import Chem
from icolos.core.containers.compound import Conformer
from icolos.utils.execute_external.execute import Executor
from icolos.core.workflow_steps.step import _LE
from icolos.core.workflow_steps.calculation.base import StepCalculationBase
from icolos.utils.general.parallelization import Parallelizer, Subtask, SubtaskContainer


class StepGenericScore(StepCalculationBase, BaseModel):
    """
    Run an arbitrary scoring script, taking one or
    more compounds as input, and producing a score.

    It assumes that the script takes a smiles string as input, and
    creates an SDF file with the scored pose and a score attached as a tag.

    Input
    -----
    :compounds source: Set of compounds from a previous step

    Settings
    --------
    :additional script_path: Path to the script to run
    :additional interpreter: Interpreter to use to run the script,
        assumes a normal binary (i.e. no interpreter) by default

    """

    _interpreter = None
    _script_path = None
    _arguments = None

    def __init__(self, **data):
        super().__init__(**data)

        self._interpreter = None
        self._script_path = None
        self._arguments = None
        self._initialize_backend(executor=Executor)
        self._check_backend_availability()

    def _run_script(self, tmp_dir: Path, sublist: list[Subtask]):
        self._logger.log(f"Running {sublist}", _LE.DEBUG)
        enum = sublist[0].data
        interpreter = self._interpreter if self._interpreter is not None else ""
        result = self._backend_executor.execute(
            command=interpreter + " " + self._script_path.as_posix(),
            arguments=[f'"{enum.get_smile()}"'] + self._arguments,
            location=tmp_dir.as_posix()
        )

    def _run_all(self):
        script_parallelizer = Parallelizer(func=self._run_script)

        while not self._subtask_container.done():
            next_batch = self._get_sublists(get_first_n_lists=self._get_number_cores())
            _ = [sub.increment_tries() for element in next_batch for sub in element]
            _ = [sub.set_status_failed() for element in next_batch for sub in element]

            self._logger.log(f"Processing batch {next_batch}", _LE.DEBUG)
            # We store the mol name in the tmp dir because we only pass a smiles to the script
            tmp_dirs = [Path(tempfile.mkdtemp(suffix=f"_{sub[0].data.get_index_string()}"))
                        for sub in next_batch]
            script_parallelizer.execute_parallel(tmp_dir=tmp_dirs, sublist=next_batch)
            self._parse_script_output(next_batch, tmp_dirs)
            self._remove_temporary(tmp_dirs)
            self._log_execution_progress()

    def _parse_script_output(self, batch: list[list[Subtask]], paths: list[Path]):
        for i, path in enumerate(paths):
            sublist = batch[i]
            sdf = [p for p in path.glob("*.sdf")]
            if len(sdf) != 1:
                raise ValueError(f"Expected to find one SDF file, found {len(sdf)}")
            for mol in Chem.SDMolSupplier(sdf[0].as_posix(), removeHs=False):
                if mol is None:
                    continue

                # Retrieve the mol name from the tmpdir name
                name = path.stem.split("_")[-1]
                for compound in self.get_compounds():
                    for enumeration in compound:
                        if enumeration.get_index_string() == name:
                            self._logger.log(f"Found molecule with name {name}", _LE.DEBUG)
                            new_conformer = Conformer(conformer=mol, conformer_id=None,
                                                      enumeration_object=enumeration)
                            enumeration.add_conformer(new_conformer, auto_update=True)

                            for task in sublist:
                                if task.data.get_index_string() == name:
                                    task.set_status_success()
                            break

    def execute(self):
        # Prepare input compounds
        enumerations = []
        for compound in self.get_compounds():
            enumerations.extend(compound.get_enumerations())
            for enumeration in compound:
                enumeration.clear_conformers()

        self._subtask_container = SubtaskContainer(
            max_tries=self.execution.failure_policy.n_tries)
        self._subtask_container.load_data(enumerations)

        # Find the script
        self._interpreter = self.settings.additional.get("interpreter")
        self._script_path = Path(self.settings.additional.get("script_path"))
        if not self._script_path.exists():
            raise ValueError(f"Can't find script under {self._script_path.as_posix()}")

        # Any number of additional flags and arguments can be passed to the script
        self._arguments = []
        if self.settings.arguments.flags:
            self._arguments.extend(self.settings.arguments.flags)
        for key, value in self.settings.arguments.parameters.items():
            self._arguments.extend([key, value])

        self._run_all()
        self._logger.log(
            f"Completed execution for {self.step_id} successfully", _LE.INFO
        )
