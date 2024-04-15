""" Setup workpath - Create the PMX directory structure and setup the calculations. """

import os

from pydantic import BaseModel

from icolos.core.containers.perturbation_map import Node
from icolos.core.workflow_steps.pmx.base import StepPMXBase
from icolos.utils.enums.step_enums import StepPMXSetupEnum
from icolos.utils.execute_external.execute import Executor
from icolos.utils.execute_external.gromacs import GromacsExecutor
from icolos.utils.general.parallelization import SubtaskContainer
from icolos.core.workflow_steps.step import _LE

_SPSE = StepPMXSetupEnum()


# These classes are based on the work of Vytautas Gapsys et al: https://github.com/deGrootLab/pmx/
class StepPMXSetup(StepPMXBase, BaseModel):
    """
    Create the directory tree structure. Requires the pmx workflow
    to be executing using the single_dir running mode. Operates on
    the perturbation map object, runs acpype on the written structures
    to produce the Gromacs-compatible itp files.

    Input
    -----
    :generic pdb: Protein apo input structure, optional
    :generic log: Mapper log file
    :generic mdp:
        Directory containing Gromacs mdp configuration files,
        using the following names: "em", "nvt", "eq", "ti"
    :compounds sdf: Library of docked compounds in SDF format, optional

    Settings
    -------------------
    :additional run_type: absolute or relative mode, {rbfe, abfe}, default = rbfe
    :additional forcefield: force field to use, default = amber99sb-star-ildn-mut.ff
    :additional water: water model, default = tip3p
    :additional charge_method:
        method used to calculate ligand charges,
        {gas, bcc, user}, default = bcc
    :additional replicas: number of individual runs to perform for each edge, default = 3

    """

    _gromacs_executor: GromacsExecutor = None

    def __init__(self, **data):
        super().__init__(**data)
        self._gromacs_executor = GromacsExecutor(
            prefix_execution=self.execution.prefix_execution
        )
        self._initialize_backend(executor=Executor)

    def execute(self):
        # sets the number of replicas to be used throughput the pmx run
        replicas = self._get_additional_setting(_SPSE.REPLICAS, default=3)
        if self.work_dir is None:
            self.work_dir = self._make_tmpdir()
            self._logger.log(f"Set workflow directory to {self.work_dir}", _LE.DEBUG)
        self._construct_perturbation_map(self.work_dir, replicas)
        # create the directory structure for subsequent calculations
        edges = self.edges
        nodes = self.nodes
        self._logger.log(f"Using edges {edges} and nodes {nodes}", _LE.DEBUG)

        # create the input directory to sit at the top level of the workdir, contains ligands,
        # mdp and protein topology files
        os.makedirs(os.path.join(self.work_dir, "input"), exist_ok=True)
        for folder in ["ligands", "mdp", "protein"]:
            os.makedirs(os.path.join(self.work_dir, "input", folder), exist_ok=True)

        # handle protein parametrization with pdb2gmx
        protein = (
            self.get_workflow_object().workflow_data.perturbation_map.get_protein()
        )
        self._logger.log(f"Using protein {protein.get_file_name()}", _LE.DEBUG)

        protein.write(os.path.join(self.work_dir, "input/protein"))
        self._logger.log(f"Wrote protein {os.path.join(self.work_dir, 'input/protein')}", _LE.DEBUG)

        self._parametrise_protein(protein=protein.get_file_name(), path="input/protein")

        # remove the backup file
        old_protein = [
            f
            for f in os.listdir(os.path.join(self.work_dir, "input/protein"))
            if f.endswith("#")
        ]
        # only want the parametrised processed pdb file in there
        old_protein.append(protein.get_file_name())
        for f in old_protein:
            os.remove(os.path.join(self.work_dir, "input/protein", f))

        self._clean_protein()

        mdp_dir = self.data.generic.get_argument_by_extension(
            ext="mdp", rtn_file_object=True
        )
        mdp_dir.write(os.path.join(self.work_dir, "input/mdp"))

        # parallelize the antechamber call across the pool of nodes

        self.execution.parallelization.max_length_sublists = 1
        self._subtask_container = SubtaskContainer(
            max_tries=self.execution.failure_policy.n_tries
        )
        self._subtask_container.load_data(nodes)
        self._execute_pmx_step_parallel(
            run_func=self._parametrise_nodes,
            step_id="pmx_setup",
            result_checker=self._check_results,
        )

        # create the output folder structure
        for edge in edges:
            edgepath = os.path.join(
                self.work_dir,
                str(f"{edge.node_from.get_node_hash()}_{edge.node_to.get_node_hash()}"),
            )
            hybridTopFolder = f"{edgepath}/hybridStrTop"
            os.makedirs(hybridTopFolder, exist_ok=True)

            # water/protein
            for wp in self.therm_cycle_branches:
                wppath = f"{edgepath}/{wp}"
                os.makedirs(wppath, exist_ok=True)

                # stateA/stateB
                for state in self.states:
                    statepath = f"{wppath}/{state}"
                    os.makedirs(statepath, exist_ok=True)

                    # run1/run2/run3
                    for r in range(1, replicas + 1):
                        runpath = f"{statepath}/run{r}"
                        os.makedirs(runpath, exist_ok=True)

                        # em/eq_posre/eq/transitions
                        for sim in self.sim_types:
                            simpath = f"{runpath}/{sim}".format(runpath, sim)
                            os.makedirs(simpath, exist_ok=True)

    def _check_results(self, batch: list[list[Node]]) -> list[list[bool]]:
        """
        Check the output of the node parametrisation.

        :param batch: nested list of nodes
        :returns results: nested list of job results

        """
        output_files = ["ffMOL.itp", "MOL.itp", "MOL.pdb"]
        results = []
        for subjob in batch:
            subjob_results = []
            for job in subjob:
                subjob_results.append(
                    all(os.path.isfile(os.path.join(
                            self.work_dir, "input/ligands", job.get_node_hash(), f))
                        for f in output_files))
            results.append(subjob_results)
        return results
