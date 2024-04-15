""" Prepare transitions - Create input files for transition simulations """

import os

from pydantic import BaseModel

from icolos.core.workflow_steps.pmx.base import StepPMXBase
from icolos.core.workflow_steps.step import _LE
from icolos.utils.enums.program_parameters import GromacsEnum
from icolos.utils.execute_external.gromacs import GromacsExecutor
from icolos.utils.general.parallelization import SubtaskContainer

_GE = GromacsEnum()


class StepPMXPrepareTransitions(StepPMXBase, BaseModel):
    """
    Prepare transitions: extract snapshots from equilibrium simulations, prepare .tpr files for each

    Requires
    --------
    :work_dir dir: Working directory containing output of previous steps, optional
    :compounds perturbation_map: Previously constructed perturbation map

    """

    def __init__(self, **data):
        super().__init__(**data)

        self._initialize_backend(executor=GromacsExecutor)

    def execute(self):
        self.execution.parallelization.max_length_sublists = 1
        self._subtask_container = SubtaskContainer(
            max_tries=self.execution.failure_policy.n_tries
        )
        self._subtask_container.load_data(self.edge_ids)
        self._execute_pmx_step_parallel(
            run_func=self.prepare_transitions,
            step_id="pmx prepare_transitions",
            result_checker=self._check_result,
        )

    def _extract_snapshots(self, eqpath: str, tipath: str):
        """
        Extract snapshots from equilibration runs to run transitions.

        :param eqpath: path to the equilibrium simulation
        :param tipath: output path for the transitions

        """
        tpr = "{0}/tpr.tpr".format(eqpath)
        trr = "{0}/traj.trr".format(eqpath)
        frame = "{0}/frame.gro".format(tipath)

        trjconv_args = {
            "-s": tpr,
            "-f": trr,
            "-o": frame,
            "-sep": "",
            "-ur": "compact",
            "-pbc": "mol",
        }
        trjconv_args = self.get_arguments(trjconv_args)
        self._backend_executor.execute(
            _GE.TRJCONV, arguments=trjconv_args, pipe_input="echo System", check=False
        )

        last_frame = len([f for f in os.listdir(tipath) if f.startswith("frame")])
        if last_frame == 0:
            raise FileNotFoundError(f"No frames extracted, are you sure the trr is long enough?")

        self._logger.log(f"Extracted {last_frame} frames", _LE.DEBUG)

        self._clean_backup_files(tipath)
        # once frames are extracted, remove the large trr file from the equilibrium run
        files_to_remove = [trr, tpr, os.path.join(eqpath, "frame.gro")]
        for f in files_to_remove:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

    def _prepare_system(self, edge: str, state: str, wp: str, r: int, toppath: str):
        """
        Prepare a single system.

        :param edge: edge ID to prepare
        :param state: endpoint state to prepare
        :param wp: thermodynamic cycle to calculate
        :param r: replica number
        :param toppath: system topology file
        
        """
        eqpath = self._get_specific_path(
            work_path=self.work_dir,
            edge=edge,
            therm_cycle=wp,
            state=state,
            replica=r,
            sim="eq",
        )
        tipath = self._get_specific_path(
            work_path=self.work_dir,
            edge=edge,
            therm_cycle=wp,
            state=state,
            replica=r,
            sim="transitions",
        )
        # if the trr file exists and snapshots have not been extracted in a previous run
        if os.path.isfile(os.path.join(eqpath, "traj.trr")) and not os.path.isfile(
            os.path.join(tipath, "frame0.gro")
        ):
            self._extract_snapshots(eqpath, tipath)
        else:
            self._logger.log("Skipping frame extraction, already present", _LE.DEBUG)
        print("preparing tpr files")

        self._prepare_single_tpr(
            simpath=tipath,
            toppath=toppath,
            state=state,
            sim_type="transitions",
            executor=self._backend_executor,
        )

        self._clean_backup_files(tipath)

    def prepare_transitions(self, jobs: list[str]):
        """
        Prepare systems for transition simulations.

        :param jobs: list of edges to run

        """
        for edge in jobs:
            ligTopPath = self._get_specific_path(
                work_path=self.work_dir, edge=edge, therm_cycle="unbound"
            )
            protTopPath = self._get_specific_path(
                work_path=self.work_dir, edge=edge, therm_cycle="bound"
            )
            for state in self.states:
                for r in range(1, self.get_perturbation_map().replicas + 1):

                    self._logger.log(
                        f"Preparing transitions: {edge}, {state}, run {r}", _LE.DEBUG
                    )
                    self._prepare_system(
                        edge=edge, state=state, wp="unbound", r=r, toppath=ligTopPath
                    )
                    self._prepare_system(
                        edge=edge, state=state, wp="bound", r=r, toppath=protTopPath
                    )

    def _check_result(self, batch: list[list[str]]) -> list[list[bool]]:
        """
        Look in each hybridStrTop dir and check the output pdb files exist for the edges

        :param batch: nested list of transitions to check
        :returns results: nested list of checking results

        """
        replicas = self.get_perturbation_map().replicas
        output_paths = []
        for i in range(1, replicas + 1):
            output_paths.append(f"unbound/stateA/run{i}/transitions")
            output_paths.append(f"unbound/stateB/run{i}/transitions")
            output_paths.append(f"bound/stateA/run{i}/transitions")
            output_paths.append(f"bound/stateB/run{i}/transitions")

        results = []
        for subjob in batch:
            subjob_results = []
            for job in subjob:
                # check number of frames extracted == number tpr files for each leg

                num_frames = [
                    len(
                        [
                            f
                            for f in os.listdir(os.path.join(self.work_dir, job, f))
                            if f.startswith("frame")
                        ]
                    )
                    for f in output_paths
                ]
                num_tprs = [
                    len(
                        [
                            f
                            for f in os.listdir(os.path.join(self.work_dir, job, f))
                            if f.startswith("ti")
                        ]
                    )
                    for f in output_paths
                ]
                self._logger.log(
                    f"Found {num_tprs} tpr files and {num_frames} frame files for edge {job}",
                    _LE.DEBUG,
                )
                # confirms that frames have been extracted, and we have a tpr file generated for each ti frame
                subjob_results.append(
                    num_tprs == num_frames and all(n > 10 for n in num_frames)
                )
            results.append(subjob_results)
        return results
