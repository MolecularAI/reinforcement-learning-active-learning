""" Base class for all PMX-related workflow steps. """

import glob
import os
import shutil
import time
from typing import Callable, Literal, Optional, Any, Union

from pydantic import BaseModel
from rdkit.Chem import rdmolops  # type: ignore
from rdkit import Chem  # type: ignore

from icolos.core.containers.compound import Compound, Conformer
from icolos.core.containers.perturbation_map import Edge, Node, PerturbationMap
from icolos.core.workflow_steps.step import StepBase
from icolos.utils.enums.logging_enums import LoggingConfigEnum
from icolos.utils.enums.parallelization import ParallelizationEnum
from icolos.utils.enums.program_parameters import GromacsEnum, SlurmEnum, StepPMXEnum
from icolos.utils.enums.step_enums import StepGromacsEnum, StepPMXSetupEnum
from icolos.utils.execute_external.execute import Executor, ExecutorBase
from icolos.utils.execute_external.gromacs import GromacsExecutor
from icolos.utils.general.progress_bar import get_progress_bar_string
from icolos.utils.general.parallelization import Parallelizer, Subtask

_LE = LoggingConfigEnum()
_GE = GromacsEnum()
_SGE = StepGromacsEnum()
_SPSE = StepPMXSetupEnum()
_SPE = StepPMXEnum()

# FIXME the way this "enum" is handled is rather messy
_PE = ParallelizationEnum
_SE = SlurmEnum()

SimType = Literal["em", "nvt", "eq", "transitions"]
EndPoint = Literal["stateA", "stateB"]
ThermCycle = Literal["unbound", "bound"]


class StepPMXBase(StepBase, BaseModel):
    """
    Base class containing shared methods for Non-equilibrium free energy calculations

    Settings
    --------

    These apply to any step that inherits StepPMXBase.

    :additional run_type: absolute or relative mode, {rbfe, abfe}, default = rbfe
    :additional forcefield: force field to use, default = amber99sb-star-ildn-mut.ff
    :additional boxshape: the boxshape to use in calculation setup, {cubic, octahedron, dodecahedron}, default = dodecahedron
    :additional boxd: distance of solute to box edge, default = 1.5
    :additional water: water model, default = tip3p
    :additional conc: salt concentration, default = 0.15
    :additional pname: cation type, default = NaJ
    :additional pname: anion type, default = ClJ
    :additional charge_method: method used to calculate ligand charges, {gas, bcc, user}, default = bcc
    :additional topology: perturbation graph topology, {normal, star}, default = normal
    :additional hub_conformer: hub conformer for perturbation star map, default = None
    :additional strict: prune failed perturbation map edges, default = True

    """

    _antechamber_executor: Optional[Executor]
    _gromacs_executor: Optional[ExecutorBase]
    sim_types: Optional[list[str]]
    states: Optional[list[str]]
    therm_cycle_branches: Optional[list[str]]
    run_type: Optional[str]
    force_field: Optional[str]
    boxshape: Optional[str]
    boxd: Optional[float]
    water: Optional[str]
    conc: Optional[float]
    pname: Optional[str]
    nname: Optional[str]
    mdp_prefixes: Optional[dict[str, str]]

    def __init__(self, **data):
        super().__init__(**data)

        self._antechamber_executor = Executor()
        self._gromacs_executor = GromacsExecutor(
            prefix_execution=self.execution.prefix_execution
        )
        self.sim_types = ["em", "nvt", "eq", "transitions"]
        self.states = ["stateA", "stateB"]
        # for a normal pmx run this would be "water" and "protein"
        # unbound -> ligand, bound -> complex
        self.therm_cycle_branches = ["unbound", "bound"]

        # simulation setup
        self.run_type = self._get_additional_setting(_SPE.RUN_TYPE, "rbfe")
        self.force_field = "amber99sb-star-ildn-mut.ff"
        self.boxshape = self._get_additional_setting(_SPE.BOXSHAPE, "dodecahedron")
        self.boxd = self._get_additional_setting(_SPE.BOXD, 1.5)
        self.water = self._get_additional_setting(_SPE.WATER, "tip3p")
        self.conc = self._get_additional_setting(_SPE.CONC, 0.15)
        self.pname = self._get_additional_setting(_SPE.PNAME, "NaJ")
        self.nname = self._get_additional_setting(_SPE.NNAME, "ClJ")
        self.mdp_prefixes = {
            "em": "em",
            "nvt": "nvt",
            "npt": "npt",
            "eq": "eq",
            "transitions": "ti",
        }

    # TODO make this a general function
    def get_arguments(self, defaults: Optional[dict[str, str]]=None) -> list[str]:
        """
        Construct pmx-specific arguments from the step defaults,
        overridden by arguments specified in the config file

        :param defaults: optional set of key-value pairs for default arguments
        :returns: formatted list of strings
        """
        arguments = []

        # add flags
        for flag in self.settings.arguments.flags:
            arguments.append(flag)

        # flatten the dictionary into a list for command-line execution
        for key, value in self.settings.arguments.parameters.items():
            arguments.append(key)
            arguments.append(value)

        # add defaults, if not already present
        if defaults is not None:
            for key, value in defaults.items():
                if key not in arguments:
                    arguments.append(key)
                    arguments.append(value)
        return arguments

    def _get_specific_path(
        self,
        work_path: Optional[str]=None,
        edge: Optional[str]=None,
        b_hybrid_str_top: bool=False,
        therm_cycle: ThermCycle=None,
        state: EndPoint=None,
        replica: Optional[str]=None,
        sim: SimType=None,
    ) -> Optional[str]:
        """
        Utility function for getting the right paths from a pmx-type directory structure. Works for both rbfe and abfe runs.

        :param work_path: current working path
        :param edge: Node-Node edge name
        :param b_hybrid_str_top: use hybrid topology
        :param therm_cycle: path including edge
        :param state: endpoint state
        :param replica: run
        :param sim: simulation type
        :returns path: full path

        """
        if edge is None:
            return work_path
        edgepath = f"{work_path}/{edge}"

        if b_hybrid_str_top:
            hybrid_str_path = f"{edgepath}/hybridStrTop"
            return hybrid_str_path

        if therm_cycle is None:
            return edgepath
        wppath = f"{edgepath}/{therm_cycle}"

        if state is None:
            return wppath
        statepath = f"{wppath}/{state}"

        if replica is None:
            return statepath
        runpath = f"{statepath}/run{replica}"

        if sim is None:
            return runpath
        simpath = f"{runpath}/{sim}"
        return simpath

    def _parametrise_protein(
        self,
        protein: str = "protein.pdb",
        path: str = "input/protein",
        output: str ="protein.pdb",
    ):
        """
        Create the topology by running pdb2gmx.

        :param protein: path to the protein
        :param path: path for the output
        :param output: output name

        """
        # run pdb2gmx on the protein
        pdb2gmx_args = [
            "-f",
            os.path.join(self.work_dir, path, protein),
            "-ignh",
            "-water",
            self.settings.additional["water"],
            "-ff",
            self.settings.additional["forcefield"],
            "-o",
            os.path.join(self.work_dir, path, output),
        ]
        self._gromacs_executor.execute(
            command=_GE.PDB2GMX,
            arguments=pdb2gmx_args,
            check=True,
            location=os.path.join(self.work_dir, path),
        )

    def _prepare_single_tpr(
        self,
        simpath: str,
        toppath: str,
        state: EndPoint,
        sim_type: SimType,
        executor: Executor,
        empath: Optional[str]=None,
    ):
        """
        Prepare an MD run by running grompp.

        :param simpath: output directory for tpr
        :param toppath: directory containing top file
        :param state: endpoint state to prepare
        :param sim_type: kind of simulation to run
        :param executor: executor instance to run grompp with
        :param empath: directory containing energy minimization output

        """
        mdp_path = os.path.join(self.work_dir, "input/mdp")
        mdp_prefix = self.mdp_prefixes[sim_type]

        # TODO: is this a liability? would we ever have more than a single topol file?
        top = f"{toppath}/*.top"
        tpr = f"{simpath}/tpr.tpr"
        mdout = f"{simpath}/mdout.mdp"
        # mdp
        if state == "stateA":
            mdp = f"{mdp_path}/{mdp_prefix}_l0.mdp"
        else:
            mdp = f"{mdp_path}/{mdp_prefix}_l1.mdp"
        # TODO: deal with nvt/npt for abfe
        if not sim_type == "transitions":
            if sim_type == "em":
                if self.run_type == "rbfe":
                    in_str = f"{toppath}/ions.pdb"
                elif self.run_type == "abfe":
                    in_str = f"{toppath}/genion.gro"
            elif sim_type in ("eq", "nvt", "npt"):
                in_str = f"{empath}/confout.gro"

            grompp_args: list[str] = [
                "-f",
                mdp,
                "-c",
                in_str,
                "-r",
                in_str,
                "-p",
                top,
                "-o",
                tpr,
                "-maxwarn",
                "4",
                "-po",
                mdout,
            ]
            if not os.path.isfile(tpr):
                _ = executor.execute(
                    command=_GE.GROMPP,
                    arguments=grompp_args,
                    check=True,
                    location=simpath,
                )
            else:
                self._logger.log(f"tpr file {tpr} already exists, skipping", _LE.DEBUG)

        elif sim_type == "transitions":
            grompp_full_cmd: list[str] = []
            # 80 frames = 0 - 79
            num_frames = len([f for f in os.listdir(simpath) if f.startswith("frame")])
            self._logger.log(
                f"Generating transition tpr files for {num_frames} frames", _LE.DEBUG
            )
            for frame in range(num_frames):
                in_str = f"{simpath}/frame{frame}.gro"
                tpr = f"{simpath}/ti{frame}.tpr"

                grompp_args = [
                    "gmx grompp",
                    "-f",
                    mdp,
                    "-c",
                    in_str,
                    "-r",
                    in_str,
                    "-p",
                    top,
                    "-o",
                    tpr,
                    "-maxwarn",
                    "4",
                    "-po",
                    mdout,
                    ";",
                ]
                if not os.path.isfile(tpr):
                    grompp_full_cmd += grompp_args
                else:
                    self._logger.log(
                        f"tpr file {tpr} already exists, skipping", _LE.DEBUG
                    )
            grompp_cmd: str = " ".join(grompp_full_cmd[:-1])
            # check all transitions have not been skipped
            if grompp_cmd:
                _ = executor.execute(
                    command=grompp_cmd, arguments=[], check=True, location=simpath
                )
        self._clean_backup_files(simpath)

    def _clean_pdb_structure(self, tmp_dir: str):
        """
        Removes problematic PDB lines that might cause problems later.

        :param tmp_dir: directory containing PDB files to be cleaned

        """
        files = [file for file in os.listdir(tmp_dir) if file.endswith("pdb")]
        for file in files:
            cleaned_lines = []
            with open(os.path.join(tmp_dir, file), "r") as f:
                lines = f.readlines()
            for line in lines:
                if "ATOM" in line or "HETATM" in line:
                    cleaned_lines.append(line)
            with open(os.path.join(tmp_dir, file), "w") as f:
                f.writelines(cleaned_lines)

    def _parametrisation_pipeline(
        self, tmp_dir: str, conf: Conformer, include_top: bool=False, include_gro: bool=False
    ):
        """
        Parametrises compounds using antechamber.

        :param tmp_dir: directory in which to perform the parametrisation
        :param conf: conformer instance to parametrise
        :param include_top: whether to also output the top file, as opposed to only the itp file
        :param include_gro: whether to also output the gro file

        """
        # main pipeline for producing GAFF parameters for a ligand
        charge_method = self._get_additional_setting(
            key=_SPSE.CHARGE_METHOD, default="bcc"
        )
        formal_charge = (
            rdmolops.GetFormalCharge(conf.get_molecule()) if conf is not None else 0
        )
        arguments_acpype = [
            "-di",
            "MOL.sdf",
            "-c",
            charge_method,
            "-a",
            "gaff2",
            "-n",
            formal_charge,
        ]
        self._logger.log("Generating ligand parameters...", _LE.DEBUG)
        self._antechamber_executor.execute(
            command=_GE.ACPYPE_BINARY,
            arguments=arguments_acpype,
            location=tmp_dir,
            check=True,
        )
        # search the output dir for the itp file
        acpype_dir = [p for p in os.listdir(tmp_dir) if p.endswith(".acpype")][0]
        self._logger.log(f"Found acpype directory at {acpype_dir}", _LE.DEBUG)
        itp_file = [
            f
            for f in os.listdir(os.path.join(tmp_dir, acpype_dir))
            if f.endswith("GMX.itp")
        ][0]
        self._logger.log(f"Found itp file at {itp_file}", _LE.DEBUG)
        pdb_file = [
            f
            for f in os.listdir(os.path.join(tmp_dir, acpype_dir))
            if f.endswith("NEW.pdb")
        ][0]
        self._logger.log(f"Found pdb file at {pdb_file}", _LE.DEBUG)
        shutil.copyfile(
            os.path.join(tmp_dir, acpype_dir, itp_file),
            # standardized name must be enforced here to make argument
            # parsing easier in subsequent pmx steps
            os.path.join(tmp_dir, "MOL.itp"),
        )
        shutil.copyfile(
            os.path.join(tmp_dir, acpype_dir, pdb_file),
            # standardized name must be enforced here to make argument
            # parsing easier in subsequent pmx steps
            os.path.join(tmp_dir, "MOL.pdb"),
        )
        # for abfe calculations we need the ligand_GMX.top + .gro files as well
        if include_top:
            top_file = [
                f
                for f in os.listdir(os.path.join(tmp_dir, acpype_dir))
                if f.endswith("GMX.top")
            ][0]
            shutil.copyfile(
                os.path.join(tmp_dir, acpype_dir, top_file),
                os.path.join(tmp_dir, top_file),
            )
        if include_gro:
            gro_file = [
                f
                for f in os.listdir(os.path.join(tmp_dir, acpype_dir))
                if f.endswith("GMX.gro")
            ][0]
            shutil.copyfile(
                os.path.join(tmp_dir, acpype_dir, gro_file),
                os.path.join(tmp_dir, gro_file),
            )

    def _run_job_pool(self, run_func: Callable[[Any], Any]):
        """
        Run all jobs in the subtask container

        :param run_func: function to run on the stored tasks

        """
        # get the loaded tasks from the subtask container
        job_generator = (j for j in self._subtask_container.get_todo_tasks())
        n_jobs = len(self._subtask_container.get_todo_tasks())
        current_jobs: list[Subtask] = []
        # initially fill the queue with N jobs
        while len(current_jobs) < self.execution.parallelization.jobs:
            try:
                current_jobs.append(next(job_generator))
            except StopIteration:
                break

        _ = [job.increment_tries() for job in current_jobs]
        # submit the initial job pool
        queue_exhausted = False
        previous_metrics = [0, 0, 0]
        done_count = 0
        while done_count < n_jobs:
            # loop through the jobs:
            done_count = len(self._subtask_container.get_done_tasks())
            running_count = len(self._subtask_container.get_running_tasks())
            ready_count = len(self._subtask_container.get_todo_tasks())

            current_metrics = [done_count, running_count, ready_count]
            if current_metrics != previous_metrics:
                self._logger.log(
                    f" Execution Summary: PENDING: {ready_count}\tRUNNING: {running_count}\tDONE: {done_count}",
                    _LE.INFO,
                )
                prog_string = get_progress_bar_string(
                    done_count, done_count + running_count + ready_count
                )
                self._logger.log(prog_string, _LE.INFO)
            previous_metrics = current_metrics
            for job in current_jobs:
                # job is ready to go, dispatch it to Slurm
                if job.status == _PE.STATUS_READY:
                    job_id = run_func(job.data)
                    job.set_job_id(job_id)
                    job.set_status(_PE.STATUS_RUNNING)
                # check the job status
                elif job.status == _PE.STATUS_RUNNING:
                    # check to see whether it's finished
                    status = self._backend_executor._check_job_status(job.job_id)
                    if status == _SE.COMPLETED:
                        self._logger.log(f"Job {job.job_id} COMPLETED", _LE.DEBUG)
                        job.set_status_success()
                    elif status == _SE.FAILED:
                        self._logger.log(f"Job {job.job_id} FAILED!", _LE.WARNING)
                        job.set_status_failed()
                    elif status == _SE.CANCELLED:
                        self._logger.log(
                            f"Job {job.job_id} was CANCELLED!", _LE.WARNING
                        )
                        job.set_status_failed()
                    elif status == _SE.NODE_FAIL:
                        # aws revoked the spot instance.  Resubmit the job
                        self._logger.log(
                            f"Job {job.job_id} was revoked, resubmitting...", _LE.DEBUG
                        )
                        job.set_status(_PE.STATUS_READY)
                    elif status not in (_SE.RUNNING, _SE.PENDING):
                        self._logger.log(
                            f"Unhandled job state {status} for job {job.job_id}",
                            _LE.WARNING,
                        )
                        job.set_status_failed()

                # if complete, succesfully or not, remove the job from the queue, prepare another
                elif job.status in (_PE.STATUS_SUCCESS, _PE.STATUS_FAILED):
                    current_jobs.remove(job)
                    if not queue_exhausted:
                        try:
                            new_job = next(job_generator)
                            self._logger.log(f"Preparing new job {job.data}", _LE.DEBUG)
                            new_job.increment_tries()
                            current_jobs.append(new_job)
                        except StopIteration:
                            self._logger.log("Reached end of job queue", _LE.DEBUG)
                            queue_exhausted = True
            time.sleep(10)

    def _execute_pmx_step_parallel(
        self,
        run_func: Callable[[Any], Any],
        step_id: str,
        result_checker: Callable[[Any], list[list[bool]]],
        prune_completed: bool = True,
        **kwargs,
    ):
        """
        Instantiates Icolos's parallelizer object, runs the step's execute method,
        passes any kwargs straight to the run_func.

        :param run_func: function to run in parallel on the stored jobs
        :param step_id: name of the current step
        :param result_checker: function to use to check run_func output for completion
        :param prune_completed: whether to remove completed jobs
        :param kwargs: additional arguments passed to the executor

        """
        parallelizer = Parallelizer(func=run_func, single=self.is_debug)
        n = 1
        while not self._subtask_container.done():

            next_batch = self._get_sublists(
                get_first_n_lists=self._get_number_cores()
            )  # return n lists of length max_sublist_length
            _ = [sub.increment_tries() for element in next_batch for sub in element]
            _ = [sub.set_status_failed() for element in next_batch for sub in element]

            jobs = self._prepare_edges(next_batch)
            n_removed = 0
            if prune_completed:
                pre_exec_results = result_checker(jobs)
                for job_sublist, exec_success_sublist, sublist in zip(
                    jobs, pre_exec_results, next_batch
                ):
                    # we test on the subtask level, not the individual job level, but since jobs are run through with max_len_sublists=1, in practice this doesn't matter
                    for job, result, task in zip(
                        job_sublist, exec_success_sublist, sublist
                    ):
                        if result:
                            # remove the entire sublist (one fewer cores running)
                            job_sublist.remove(job)
                            task.set_status_success()
                            self._logger.log(
                                f"Removed job {job} from execution batch, good output found",
                                _LE.DEBUG,
                            )
                            n_removed += 1
                        # if we have emptied entire job queues, remove the queue
                self._logger.log(
                    f"Executing {step_id} for batch {n}, containing {len(jobs)} * {self.execution.parallelization.max_length_sublists} jobs",
                    _LE.INFO,
                )

            jobs = [j for j in jobs if j]
            parallelizer.execute_parallel(jobs=jobs, **kwargs)

            self._logger.log("Checking execution results...", _LE.DEBUG)
            batch_results = result_checker(jobs)
            good_results = 0
            for tasks, results in zip(next_batch, batch_results):
                # returns boolean arrays: False => failed job
                for subtask, sub_result in zip(tasks, results):
                    if not sub_result:
                        subtask.set_status_failed()
                        self._logger.log(f"Warning: job {subtask} failed!", _LE.WARNING)
                        if (
                            self.get_perturbation_map() is not None
                            and self.get_perturbation_map().strict_execution
                            and isinstance(subtask.data, str)
                        ):
                            edge = self.get_perturbation_map().get_edge_by_id(
                                subtask.data
                            )
                            if edge is not None:
                                edge.set_status(_PE.STATUS_FAILED)

                    else:
                        subtask.set_status_success()
                        good_results += 1

            self._logger.log(
                f"EXECUTION SUMMARY: Completed {good_results} jobs successfully (out of {len(next_batch) * len(next_batch[0])} jobs for step {step_id}. Removed {n_removed} already completed jobs",
                _LE.INFO,
            )

            self._log_execution_progress()
            n += 1

    @property
    def edges(self) -> list[Edge]:
        """
        Inspect the map object passed to the step and extract the edge info.

        """
        return self.get_workflow_object().workflow_data.perturbation_map.edges

    @property
    def nodes(self) -> list[Node]:
        """
        Return the nodes attached to the perturbation map.

        """
        return self.get_workflow_object().workflow_data.perturbation_map.nodes

    @property
    def edge_ids(self) -> list[str]:
        """
        Return the ids of the perturbation map's edges.

        """
        return [edge.get_edge_id() for edge in self.edges]

    def _get_line_idx(self, data: list[str], id_str: str) -> int:
        """
        Find the index of a line containing a string.

        :param data: lines to search
        :param id_str: search string
        :returns index: index of found line

        """
        line = [e for e in data if id_str in e]
        assert len(line) == 1
        target = line[0]
        return data.index(target)

    def _clean_protein(self):
        """
        Extracts a chain from a topology if required.

        """
        existing_itp_files = [f for f in os.listdir(os.path.join(self.work_dir, "input/protein"))
                              if f.endswith("itp") and "Protein" in f]
        if not existing_itp_files:  # no protein itp files, we have a single chain that needs extacting from the top file
            with open(os.path.join(self.work_dir, "input/protein/topol.top"), "r") as f:
                top_lines = f.readlines()

            moltype_line = self._get_line_idx(top_lines, _GE.MOLECULETYPES)
            end_itp_line = self._get_line_idx(top_lines, "; Include water topology")

            moltype = top_lines[moltype_line + 2].split()[0]
            cleaned_top = (
                top_lines[:moltype_line]
                + [f'#include "topol_{moltype}.itp']
                + top_lines[end_itp_line:]
            )

            itp_lines = top_lines[moltype_line:end_itp_line]

            with open(os.path.join(self.work_dir, "input/protein/topol.top"), "w") as f:
                f.writelines(cleaned_top)

            with open(os.path.join(self.work_dir, f"input/protein/topol_{moltype}.itp"), "w") as f:
                f.writelines(itp_lines)

    def get_hub_conformer(self, hub_conf_path: str) -> Conformer:
        """
        Creates a conformer representing the hub of the map, from an sdf file.

        :param hub_conf_path: path to the hub conformer SDF file
        :results conformer: conformer instance

        """
        with Chem.SDMolSupplier(hub_conf_path) as supplier:
            hub_mol = supplier[0]
        return Conformer(conformer=hub_mol)

    def _construct_perturbation_map(self, work_dir: str, replicas: int):
        """
        Constructs a perturbation map for binding free energy calculations.

        :param work_dir: path to write output data to
        :param replicas: number of run replicas (?)

        """

        if self.get_perturbation_map() is not None:
            self._logger.log("Perturbation map already constructed", _LE.DEBUG)
            self.get_perturbation_map().protein = (
                self.data.generic.get_argument_by_extension("pdb", rtn_file_object=True)
            )
            self.get_perturbation_map().replicas = replicas
            return
        topology = self._get_additional_setting("topology", default="normal")
        # check whether a hub conformer has been supplied (as an sdf file)
        hub_conf_path = self._get_additional_setting("hub_conformer", default=None)

        if hub_conf_path is not None:
            assert hub_conf_path.endswith(
                ".sdf"
            ), "Hub conformer must be supplied as an SDF file!"

        perturbation_map = PerturbationMap(
            compounds=self.data.compounds,
            protein=self.data.generic.get_argument_by_extension(
                "pdb", rtn_file_object=True
            ),
            replicas=replicas,
            strict_execution=self._get_additional_setting(_SPE.STRICT, default=True),
            hub_conformer=self.get_hub_conformer(hub_conf_path)
            if hub_conf_path is not None
            else None,
        )
        if topology == "normal":
            # construct the perturbation map and load in the log file
            log_file = self.data.generic.get_argument_by_extension(
                "log", rtn_file_object=True
            )
            log_file.write(work_dir)

            perturbation_map.parse_map_file(
                os.path.join(self.work_dir, log_file.get_file_name())
            )
        elif topology == "star":
            # manually generate star top, no mapping tool required
            perturbation_map.generate_star_map()

        self._logger.log(
            f"Initialised perturbation map with {len(perturbation_map.get_nodes())} nodes and {len(perturbation_map.get_edges())} edges",
            _LE.INFO,
        )
        self.get_workflow_object().set_perturbation_map(perturbation_map)

    def _prepare_edges(self, batch: list[list[Subtask]]) -> list[list[str]]:
        """
        Prepare edges for running RBFE.

        :param batch: nested lists of subtasks to prepare
        :returns edges: nested lists of edge names

        """
        edges = []

        for task in batch:
            task_edges = []
            for element in task:
                task_edges.append(element.data)
            edges.append(task_edges)
        return edges

    def _clean_backup_files(self, path: str):
        """
        Remove superfluous backup files.

        :param path: path in which to remove files

        """
        toclean = glob.glob("{0}/*#".format(path))
        for clean in toclean:
            os.remove(clean)

    def _separate_atomtypes(self, lig_path: str):
        """
        Transfer atom types into a separate ITP file.
        
        :param lig_path: path to the ligand data
        
        """
        with open(os.path.join(lig_path, "MOL.itp"), "r") as f:
            itp_lines = f.readlines()

        start_idx = self._get_line_idx(itp_lines, _GE.ATOMTYPES)
        stop_index = self._get_line_idx(itp_lines, _GE.MOLECULETYPES)

        atomtype_lines = itp_lines[start_idx:stop_index]
        cleaned_itp_lines = itp_lines[stop_index:]
        with open(os.path.join(lig_path, "MOL.itp"), "w") as f:
            f.writelines(cleaned_itp_lines)

        # process the atomtype lines to remove the bondtype
        # col causes gmx to complain
        cleaned_atomtype_lines = []
        for line in atomtype_lines:
            parts = line.split()
            if len(parts) > 5:
                cleaned_parts = [parts[0]] + parts[2:] + ["\n"]
                cleaned_atomtype_lines.append(" ".join(cleaned_parts))
        with open(os.path.join(lig_path, "ffMOL.itp"), "w") as f:
            f.writelines(cleaned_atomtype_lines)

    def _parametrise_nodes(self, jobs: Union[list[Union[Node, Compound]], Node, Compound]):
        """
        Parametrise a compound node using antechamber.

        :param jobs: jobs in the form of nodes or compounds to be parametrised

        """
        if isinstance(jobs, list):
            node = jobs[0]
        else:
            node = jobs
        if isinstance(node, Node):
            node_id = node.get_node_hash()
            conf = node.conformer
        elif isinstance(node, Compound):
            # in abfe we pass compounds here not edges
            node_id = node.get_index_string()
            conf = node.get_enumerations()[0].get_conformers()[0]
        else:
            raise NotImplementedError(f"Cannot parametrize object of type {type(node)}")
        lig_path = os.path.join(self.work_dir, "input", "ligands", node_id)
        os.makedirs(lig_path, exist_ok=True)
        conf.write(os.path.join(lig_path, "MOL.sdf"))

        self._logger.log(f"Running parametrisation for jobs {jobs}", _LE.DEBUG)
        # now run ACPYPE on the ligand to produce the topology file
        self._parametrisation_pipeline(lig_path, conf=conf)

        self._logger.log(f"Running atom type separation for jobs {jobs}", _LE.DEBUG)
        # produces MOL.itp, need to separate the atomtypes directive out into ffMOL.itp for pmx
        # to generate the forcefield later
        self._separate_atomtypes(lig_path)
