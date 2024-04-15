""" Prepare simulations - Prepare run files for an FEP simulation """

import os
from typing import Union, Optional

from pydantic import BaseModel

from icolos.core.containers.compound import Compound
from icolos.core.containers.perturbation_map import Edge
from icolos.core.workflow_steps.pmx.base import StepPMXBase
from icolos.utils.enums.program_parameters import StepPMXEnum
from icolos.utils.execute_external.gromacs import GromacsExecutor
from icolos.utils.general.parallelization import SubtaskContainer

_PSE = StepPMXEnum()


class StepPMXPrepareSimulations(StepPMXBase, BaseModel):
    """
    Prepare the tpr file for either equilibration or production simulations

    Requires
    --------
    :work_dir dir: Working directory containing output of previous steps, optional
    :compounds perturbation_map: Previously constructed perturbation map

    Settings
    --------
    :additional sim_type: simulation type, {"em", "nvt", "eq", "transitions"}
    :additional previous_step: previous simulation to use, if not given will infer from the current step, {"em", "nvt", "eq", "transitions"}

    """

    def __init__(self, **data):
        super().__init__(**data)

        self._initialize_backend(executor=GromacsExecutor)

    def execute(self):
        if self.run_type == _PSE.RBFE:
            edges = self.edge_ids
        elif self.run_type == _PSE.ABFE:
            edges = [c.get_index_string() for c in self.get_compounds()]
        self.execution.parallelization.max_length_sublists = 1
        self._subtask_container = SubtaskContainer(
            max_tries=self.execution.failure_policy.n_tries
        )
        self._subtask_container.load_data(edges)
        self._execute_pmx_step_parallel(
            run_func=self.prepare_simulation,
            step_id="pmx prepare_simulations",
            result_checker=self._check_result,
        )

    def prepare_simulation(self, jobs: list[Union[Edge, Compound]]):
        """
        Prepare simulation inputs for RBFE or ABFE.

        :param jobs: list of edges (RBFE) or nodes (ABFE)

        """
        # define some constants that depend on whether this is rbfe/abfe
        # for abfe, edge refers to the ligand index
        sim_type = self.settings.additional[_PSE.SIM_TYPE]
        # FIXME: how do we get the replicas for abfe jobs without requiring input every time? inspect the workdir?
        replicas = (
            self.get_perturbation_map().replicas
            if self.get_perturbation_map() is not None
            else 1
        )

        for edge in jobs:
            for state in self.states:
                for r in range(1, replicas + 1):
                    for wp in self.therm_cycle_branches:
                        toppath = self._get_specific_path(
                            work_path=self.work_dir, edge=edge, therm_cycle=wp
                        )
                        # dir for the current sim type
                        simpath = self._get_specific_path(
                            work_path=self.work_dir,
                            edge=edge,
                            therm_cycle=wp,
                            state=state,
                            replica=r,
                            sim=sim_type,
                        )
                        # dir for the previous sim type, from which we get confout.gro
                        prev_type = self._get_previous_sim_type(sim_type)
                        empath = self._get_specific_path(
                            work_path=self.work_dir,
                            edge=edge,
                            therm_cycle=wp,
                            state=state,
                            replica=r,
                            sim=prev_type,
                        )
                        self._prepare_single_tpr(
                            simpath=simpath,
                            toppath=toppath,
                            state=state,
                            sim_type=sim_type,
                            empath=empath,
                            executor=self._backend_executor,
                        )

    def _get_previous_sim_type(self, sim_type: str) -> Optional[str]:
        """
        Works out where to get starting structure based on the current run and simulation type.

        :param sim_type: the type of simulation run, {"em", "nvt", "npt", "eq"}
        :returns prev_sim_type: type of previous simulation, {"em", "nvt", "npt"}

        """
        if self._get_additional_setting(_PSE.PREV_STEP) is not None:
            return self._get_additional_setting(_PSE.PREV_STEP)
        elif self.run_type == _PSE.RBFE:
            if sim_type == "nvt":
                return "em"
            elif sim_type == "eq":
                return "nvt"
        elif self.run_type == _PSE.ABFE:
            if sim_type in ("em", "nvt"):
                return "em"
            elif sim_type == "npt":
                return "nvt"
            elif sim_type == "eq":
                return "npt"
        return None

    def _check_result(self, batch: list[list[str]]) -> list[list[bool]]:
        """
        Look in each hybridStrTop dir and check if the
        output pdb files exist for the corresponding edges.

        :param batch: nested list of edges
        :returns results: nested list of successes

        """
        sim_type = self.settings.additional[_PSE.SIM_TYPE]
        replicas = self.get_perturbation_map().replicas
        output_files = []
        for i in range(1, replicas + 1):
            output_files.append(f"unbound/stateA/run{i}/{sim_type}/tpr.tpr")
            output_files.append(f"unbound/stateB/run{i}/{sim_type}/tpr.tpr")
            output_files.append(f"bound/stateA/run{i}/{sim_type}/tpr.tpr")
            output_files.append(f"bound/stateB/run{i}/{sim_type}/tpr.tpr")

        results = []
        for subjob in batch:
            subjob_results = []
            for job in subjob:
                subjob_results.append(
                    all(os.path.isfile(os.path.join(self.work_dir, job, f))
                        for f in output_files))
            results.append(subjob_results)
        return results
