""" box-water-ions - Box, solvate, and add ions to systems. """

import os

from pydantic import BaseModel

from icolos.core.workflow_steps.pmx.base import StepPMXBase
from icolos.utils.enums.program_parameters import GromacsEnum, StepPMXEnum
from icolos.utils.execute_external.pmx import PMXExecutor
from icolos.utils.general.parallelization import SubtaskContainer

_PSE = StepPMXEnum()
_GE = GromacsEnum()


class StepPMXBoxWaterIons(StepPMXBase, BaseModel):
    """
    Take the prepared structure files and prepare the system,
    runs editconf, solvate, genion and grompp for each system
    to be simulated

    Input
    -----
    :generic pdb: Protein apo input structure, optional
    :compounds sdf: Library of docked compounds in SDF format, optional

    Requires
    --------
    :compounds perturbation_map: Previously constructed perturbation map
    
    Settings
    --------
    :additional boxshape: the boxshape to use in calculation setup, {cubic, octahedron, dodecahedron}, default = dodecahedron
    :additional boxd: distance of solute to box edge, default = 1.5
    :additional conc: salt concentration, default = 0.15
    :additional pname: cation type, default = NaJ
    :additional pname: anion type, default = ClJ

    """

    def __init__(self, **data):
        super().__init__(**data)

        self._initialize_backend(executor=PMXExecutor)

    def execute(self):
        self.execution.parallelization.max_length_sublists = 1
        self._subtask_container = SubtaskContainer(
            max_tries=self.execution.failure_policy.n_tries
        )
        self._subtask_container.load_data(self.edge_ids)
        self._execute_pmx_step_parallel(
            run_func=self.boxWaterIons,
            step_id="pmx boxWaterIons",
            result_checker=self._check_result,
        )

    def boxWaterIons(self, jobs: list[str]):
        """
        Run gromacs to create a box, and add a solvent and ions.

        :param jobs: edges to prepare

        """
        mdp_path = os.path.join(self.work_dir, "input/mdp")

        for edge in jobs:
            out_lig_path = self._get_specific_path(
                work_path=self.work_dir, edge=edge, therm_cycle="unbound"
            )
            out_prot_path = self._get_specific_path(
                work_path=self.work_dir, edge=edge, therm_cycle="bound"
            )

            # box ligand
            in_str = f"{out_lig_path}/init.pdb"
            out_str = f"{out_lig_path}/box.pdb"
            editconf_args = [
                "-f",
                in_str,
                "-o",
                out_str,
                "-bt",
                self.boxshape,
                "-d",
                self.boxd,
            ]
            self._gromacs_executor.execute(
                command=_GE.EDITCONF, arguments=editconf_args, check=True
            )
            # box protein
            in_str = f"{out_prot_path}/init.pdb"
            out_str = f"{out_prot_path}/box.pdb"
            editconf_args = [
                "-f",
                in_str,
                "-o",
                out_str,
                "-bt",
                self.boxshape,
                "-d",
                self.boxd,
            ]
            self._gromacs_executor.execute(
                command=_GE.EDITCONF, arguments=editconf_args, check=True
            )
            # water ligand
            in_str = f"{out_lig_path}/box.pdb"
            out_str = f"{out_lig_path}/water.pdb"
            top = f"{out_lig_path}/topol.top"
            solvate_args = ["-cp", in_str, "-cs", "spc216.gro", "-p", top, "-o", out_str]
            self._gromacs_executor.execute(
                command=_GE.SOLVATE,
                arguments=solvate_args,
                check=True,
                location=self.work_dir,
            )
            # water protein
            in_str = f"{out_prot_path}/box.pdb"
            out_str = f"{out_prot_path}/water.pdb"
            top = f"{out_prot_path}/topol.top"
            solvate_args = ["-cp", in_str, "-cs", "spc216.gro", "-p", top, "-o", out_str]
            self._gromacs_executor.execute(
                command=_GE.SOLVATE, arguments=solvate_args, location=self.work_dir
            )

            # ions ligand
            in_str = f"{out_lig_path}/water.pdb"
            out_str = f"{out_lig_path}/ions.pdb"
            mdp = f"{mdp_path}/em_l0.mdp"
            tpr = f"{out_lig_path}/tpr.tpr"
            top = f"{out_lig_path}/topol.top"
            mdout = f"{out_lig_path}/mdout.mdp"
            grompp_args = [
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
                4,
                "-po",
                mdout,
            ]

            self._gromacs_executor.execute(
                command=_GE.GROMPP,
                arguments=grompp_args,
                check=True,
                location=self.work_dir,
            )
            genion_args = [
                "-s",
                tpr,
                "-p",
                top,
                "-o",
                out_str,
                "-conc",
                self.conc,
                "-pname",
                self.pname,
                "-nname",
                self.nname,
                "-neutral",
            ]
            self._gromacs_executor.execute(
                command=_GE.GENION,
                arguments=genion_args,
                check=True,
                pipe_input="echo SOL",
                location=self.work_dir,
            )
            # ions protein
            in_str = f"{out_prot_path}/water.pdb"
            out_str = f"{out_prot_path}/ions.pdb"
            mdp = f"{mdp_path}/em_l0.mdp"
            tpr = f"{out_prot_path}/tpr.tpr"
            top = f"{out_prot_path}/topol.top"
            mdout = f"{out_prot_path}/mdout.mdp"
            grompp_args = [
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
                4,
                "-po",
                mdout,
            ]

            self._gromacs_executor.execute(
                command=_GE.GROMPP,
                arguments=grompp_args,
                check=True,
                location=self.work_dir,
            )
            genion_args = [
                "-s",
                tpr,
                "-p",
                top,
                "-o",
                out_str,
                "-conc",
                self.conc,
                "-pname",
                self.pname,
                "-nname",
                self.nname,
                "-neutral",
            ]
            self._gromacs_executor.execute(
                command=_GE.GENION,
                arguments=genion_args,
                check=True,
                pipe_input="echo SOL",
                location=self.work_dir,
            )

            # clean backed files
            self._clean_backup_files(out_lig_path)
            self._clean_backup_files(out_prot_path)

    def _check_result(self, batch: list[list[str]]) -> list[list[bool]]:
        """
        Look in each hybridStrTop dir and check the output pdb files exist for the edges.

        :param batch: nested list of edges to check
        :returns results: nested list of checking results

        """
        output_files = [f"{f}/tpr.tpr" for f in self.therm_cycle_branches]
        results = []
        for subjob in batch:
            subjob_results = []
            for job in subjob:
                subjob_results.append(
                    all(os.path.isfile(os.path.join(self.work_dir, job, f))
                        for f in output_files))
            results.append(subjob_results)
        return results
