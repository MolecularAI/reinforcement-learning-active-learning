""" Atom mapping - hybrid topology generation for non-equilibrium switching """

import os

from pydantic import BaseModel

from icolos.core.workflow_steps.pmx.base import StepPMXBase
from icolos.utils.execute_external.pmx import PMXExecutor
from icolos.utils.enums.program_parameters import PMXEnum, PMXAtomMappingEnum
from icolos.utils.general.parallelization import SubtaskContainer

_PE = PMXEnum()
_PAE = PMXAtomMappingEnum()


class StepPMXatomMapping(StepPMXBase, BaseModel):
    """
    Ligand alchemy: map atoms for morphing.
    
    Input
    -----
    :work_dir dir: Working directory containing output of previous steps, optional

    Requires
    --------
    :compounds perturbation_map: Previously constructed perturbation map
    
    """

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_backend(executor=PMXExecutor)

    def _prepare_arguments(self, args: dict[str, str], output_dir: str) -> list[str]:
        """
        Prepare the arguments for the atom mapping step.

        :param args: additional arguments
        :param output_dir: output directory for the atom mapping
        :returns prepared_args: list of prepared arguments

        """
        # prepare the final set of arguments as a list
        prepared_args = []
        default_args = {
            "-o1": f"{output_dir}/pairs1.dat",
            "-o2": f"{output_dir}/pairs2.dat",
            "-opdb1": f"{output_dir}/out_pdb1.pdb",
            "-opdb2": f"{output_dir}/out_pdb2.pdb",
            "-opdbm1": f"{output_dir}/out_pdbm1.pdb",
            "-opdbm2": f"{output_dir}/out_pdbm2.pdb",
            "-score": f"{output_dir}/score.dat",
            "-log": f"{output_dir}/mapping.log",
        }
        default_args.update(args)

        for key, value in default_args.items():
            prepared_args.append(key)
            prepared_args.append(value)
        return prepared_args

    def _execute_command(self, jobs: list[str]):
        """
        Runs the atom mapping command for a single edge.

        :param jobs: edge to run

        """
        edge = jobs[0]
        parts = edge.split("_")
        lig1 = parts[0]
        lig2 = parts[1]
        # write them to the right dir as a pdb from the outset
        arguments = {
            "-i1": os.path.join(
                self.work_dir,
                _PAE.LIGAND_DIR,
                lig1,
                "MOL.pdb",
            ),
            "-i2": os.path.join(
                self.work_dir,
                _PAE.LIGAND_DIR,
                lig2,
                "MOL.pdb",
            ),
        }
        output_dir = os.path.join(self.work_dir, edge, _PE.HYBRID_STR_TOP)
        args = self._prepare_arguments(args=arguments, output_dir=output_dir)

        self._backend_executor.execute(
            command=_PE.ATOMMAPPING,
            arguments=args,
            check=True,
            location=self.work_dir,
        )

    def execute(self):
        # check the workflow has been configured correctly to use a shared work_dir
        assert self.work_dir is not None and os.path.isdir(self.work_dir)

        # enforce single edge per job queue
        self.execution.parallelization.max_length_sublists = 1
        self._subtask_container = SubtaskContainer(
            max_tries=self.execution.failure_policy.n_tries
        )
        self._subtask_container.load_data(self.edge_ids)
        self._execute_pmx_step_parallel(
            run_func=self._execute_command,
            step_id="atomMapping",
            result_checker=self._check_result,
        )

    def _check_result(self, batch: list[list[str]]) -> list[list[bool]]:
        """
        Look in each hybridStrTop dir and check the output pdb files exist for the edges.

        :param batch: nested list of edges
        :returns results: nested list of successes

        """
        output_files = ["pairs1.dat", "pairs2.dat", "out_pdb1.pdb", "out_pdb2.pdb"]
        results = []
        for subjob in batch:
            subjob_results = []
            for job in subjob:
                subjob_results.append(
                    all(os.path.isfile(
                            os.path.join(self.work_dir, job, "hybridStrTop", f))
                            for f in output_files))
            results.append(subjob_results)
        return results
