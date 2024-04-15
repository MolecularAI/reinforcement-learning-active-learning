from typing import List
from icolos.core.containers.compound import Conformer
from icolos.core.workflow_steps.step import StepBase
from pydantic import BaseModel

from icolos.utils.enums.program_parameters import SchrodingerExecutablesEnum
from icolos.utils.execute_external.schrodinger import SchrodingerExecutor
from icolos.utils.general.parallelization import Parallelizer
import pandas as pd
import os
from icolos.core.workflow_steps.step import _LE


_SEE = SchrodingerExecutablesEnum


class StepProteinInteraction(StepBase, BaseModel):
    def __init__(self, **data):
        super().__init__(**data)

        self._backend_executor = SchrodingerExecutor(
            prefix_execution=self.execution.prefix_execution,
            binary_location=self.execution.binary_location,
        )

    def _compute_interactions(self, conf: Conformer):
        tmp_dir = self._make_tmpdir()
        conf.write(os.path.join(tmp_dir, "structure.sdf"), format_="pdb")
        command = _SEE.PROTEIN_INTERACTION
        args = self.get_arguments(
            defaults={"-outfile": "output.csv", "-structure": "structure.pdb"}
        )

        self._backend_executor.execute(
            command=command, arguments=args, check=True, location=tmp_dir
        )

        self._parse_output(tmp_dir, conf)

    def _parse_output(self, path: str, conf: Conformer):
        # parse the output file, attach the boolean to the conformer object
        df = pd.read_csv(os.path.join(path, "output.csv"))

        # attach full interaction profile
        conf.add_extra_data("interaction_summary", df)

    def _penalize_docking_score(self, conf: Conformer, penalty: float):
        # take the docking score, add a penalty
        docking_score = conf.get_molecule().GetProp("docking_score")
        conf.get_molecule().SetProp(
            "docking_score", str(float(docking_score) + penalty)
        )

    def execute(self):
        """Runs schrodinger's protein_interaction_analysis script"""
        # requies structure file + group 1/2 identifications

        all_confs = self._unroll_compounds()

        # attach the interaction information to the conformer
        for conf in all_confs:
            self._compute_interactions(conf)

        # attach modified docking score if specific interaction is absent
        penalty = self._get_additional_setting("penalty", default=1.0)
        for conf in all_confs:
            df = conf.get_extra_data()["interaction_summary"]
            # penalize for every interaction that is not met
            base = self._get_additional_setting("base_residue")
            for interaction in self._get_additional_setting("target_residues"):
                interact_summary = df.loc[df["Residue"].str.contains(base)][
                    "Specific Interactions"
                ]
                self._logger_blank.log(interact_summary, _LE.INFO)
                try:
                    if not f"hb to {interaction}" in interact_summary.values[0]:
                        self._logger.log(
                            f"Penalizing docking score for conf {conf.get_index_string()}",
                            _LE.DEBUG,
                        )
                        self._logger.log(
                            f"Penalizing docking score for conf {conf.get_index_string()}",
                            _LE.DEBUG,
                        )

                        self._penalize_docking_score(conf, penalty)
                except Exception as e:
                    self._logger.log(e, _LE.ERROR)
                    # either no interaction summary, or something else went wrong
                    continue
