from glob import glob
import unittest
import pytest
import os
from icolos.core.workflow_steps.pmx.prepare_transitions import StepPMXPrepareTransitions
from icolos.core.composite_agents.workflow import WorkFlow
from icolos.core.workflow_steps.pmx.run_simulations import StepPMXRunSimulations
from icolos.utils.enums.step_enums import StepBaseEnum
from tests.tests_paths import (
    PATHS_EXAMPLEDATA,
    create_test_dir,
    MAIN_CONFIG,
    export_unit_test_env_vars,
    get_ligands_as_compounds_with_conformers,
)
from icolos.utils.general.files_paths import attach_root_path
from icolos.core.containers.perturbation_map import PerturbationMap

_SBE = StepBaseEnum


class Test_PMXPrepareTransitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._test_dir = attach_root_path("tests/junk/pmx/prepare_transitions")
        create_test_dir(PATHS_EXAMPLEDATA.PREPARE_TRANSITIONS_TEST_DIR, cls._test_dir)
        export_unit_test_env_vars()

    def setUp(self):
        self.compounds = get_ligands_as_compounds_with_conformers(
            PATHS_EXAMPLEDATA.PMX_TNKS_LIGANDS
        )
        p_map = PerturbationMap(compounds=self.compounds)
        p_map.parse_map_file(file_path=PATHS_EXAMPLEDATA.PMX_TNKS_MAP)
        p_map.replicas = 1
        self.p_map = p_map

    @pytest.mark.xfail(reason="had to replace missing trr trajectories with ones converted from xtcs, but now the velocities are missing, so the created frame.gro files are too small, causing the test to fail")
    def test_pmx_prepare_and_run_transitions(self):

        step_conf = {
            _SBE.STEPID: "prepare_simulations",
            _SBE.STEP_TYPE: "pmx_prepare_simulations",
            _SBE.EXEC: {
                _SBE.EXEC_PREFIXEXECUTION: MAIN_CONFIG["PMX"]["MODULE"],
                _SBE.EXEC_PARALLELIZATION: {
                    _SBE.EXEC_PARALLELIZATION_CORES: 8,
                    _SBE.EXEC_PARALLELIZATION_MAXLENSUBLIST: 1,
                },
            },
            _SBE.SETTINGS: {
                _SBE.SETTINGS_ARGUMENTS: {
                    _SBE.SETTINGS_ARGUMENTS_FLAGS: [],
                    _SBE.SETTINGS_ARGUMENTS_PARAMETERS: {
                        "-b": "2100",
                    },
                },
                _SBE.SETTINGS_ADDITIONAL: {
                    "sim_type": "transitions",
                },
            },
        }

        step_prep_trans = StepPMXPrepareTransitions(**step_conf)
        step_prep_trans.work_dir = self._test_dir
        step_prep_trans._workflow_object = WorkFlow()
        step_prep_trans._workflow_object.workflow_data.perturbation_map = self.p_map
        step_prep_trans.execute()

        stat_inf = os.stat(
            os.path.join(
                self._test_dir,
                "0ec09ef_4afa8f9/bound/stateA/run1/transitions/frame1.gro",
            )
        )
        self.assertGreater(stat_inf.st_size, 2500000)

        stat_inf = os.stat(
            os.path.join(
                self._test_dir,
                "0ec09ef_4afa8f9/bound/stateB/run1/transitions/frame1.gro",
            )
        )
        self.assertGreater(stat_inf.st_size, 2500000)

        stat_inf = os.stat(
            os.path.join(
                self._test_dir,
                "0ec09ef_4afa8f9/unbound/stateA/run1/transitions/frame1.gro",
            )
        )
        self.assertGreater(stat_inf.st_size, 357800)

        stat_inf = os.stat(
            os.path.join(
                self._test_dir,
                "0ec09ef_4afa8f9/unbound/stateB/run1/transitions/frame1.gro",
            )
        )
        self.assertGreater(stat_inf.st_size, 357800)

        step_conf = {
            _SBE.STEPID: "prepare_simulations",
            _SBE.STEP_TYPE: "pmx_prepare_simulations",
            _SBE.EXEC: {
                "platform": "slurm",
                "resources": {
                    "partition": "gpu",
                    "gres": "gpu:1",
                    "modules": [
                        "GROMACS/2021-fosscuda-2019a-PLUMED-2.7.1-Python-3.7.2"
                    ],
                },
            },
            _SBE.SETTINGS: {
                _SBE.SETTINGS_ARGUMENTS: {
                    _SBE.SETTINGS_ARGUMENTS_FLAGS: [],
                    _SBE.SETTINGS_ARGUMENTS_PARAMETERS: {},
                },
                _SBE.SETTINGS_ADDITIONAL: {"sim_type": "transitions"},
            },
        }

        step_run_simulations = StepPMXRunSimulations(**step_conf)
        step_run_simulations.work_dir = self._test_dir
        step_run_simulations._workflow_object = WorkFlow()
        step_run_simulations.get_workflow_object().workflow_data.perturbation_map = (
            self.p_map
        )
        step_run_simulations.execute()

        stat_inf = os.stat(
            glob(
                os.path.join(
                    self._test_dir,
                    "0ec09ef_4afa8f9/bound/stateB/run1/transitions/*.sh",
                )
            )[0]
        )
        self.assertEqual(stat_inf.st_size, 42290)

        stat_inf = os.stat(
            os.path.join(
                self._test_dir, "0ec09ef_4afa8f9/unbound/stateB/run1/transitions/md.log"
            )
        )

        self.assertGreater(stat_inf.st_size, 1339800)

        stat_inf = os.stat(
            os.path.join(
                self._test_dir,
                "0cd4b47_4f2ffa1/bound/stateB/run1/transitions/dhdl1.xvg",
            )
        )
        self.assertEqual(stat_inf.st_size, 72216)

    # def test_pmx_prepare_transitions_slurm(self):

    #     step_conf = {
    #         _SBE.STEPID: "prepare_simulations",
    #         _SBE.STEP_TYPE: "pmx_prepare_simulations",
    #         _SBE.EXEC: {
    #             "platform": "slurm",
    #             "resources": {
    #                 "partition": "gpu",
    #                 "gres": "gpu:1",
    #                 "modules": [
    #                     "GROMACS/2021-fosscuda-2019a-PLUMED-2.7.1-Python-3.7.2"
    #                 ],
    #             },
    #             _SBE.EXEC_PARALLELIZATION: {
    #                 _SBE.EXEC_PARALLELIZATION_CORES: 8,
    #                 _SBE.EXEC_PARALLELIZATION_MAXLENSUBLIST: 1,
    #             },
    #         },
    #         _SBE.SETTINGS: {
    #             _SBE.SETTINGS_ARGUMENTS: {
    #                 _SBE.SETTINGS_ARGUMENTS_FLAGS: [],
    #                 _SBE.SETTINGS_ARGUMENTS_PARAMETERS: {
    #                     "-b": "2100",
    #                 },
    #             },
    #             _SBE.SETTINGS_ADDITIONAL: {
    #                 "sim_type": "transitions",
    #             },
    #         },
    #     }

    #     step_prep_trans = StepPMXPrepareTransitions(**step_conf)
    #     step_prep_trans.work_dir = self._test_dir
    #     step_prep_trans._workflow_object = WorkFlow()
    #     step_prep_trans._workflow_object.workflow_data.perturbation_map = self.p_map
    #     step_prep_trans.execute()

    #     stat_inf = os.stat(
    #         os.path.join(
    #             self._test_dir,
    #             "0ec09ef_4afa8f9/bound/stateA/run1/transitions/frame1.gro",
    #         )
    #     )
    #     self.assertGreater(stat_inf.st_size, 2500000)

    #     stat_inf = os.stat(
    #         os.path.join(
    #             self._test_dir,
    #             "0ec09ef_4afa8f9/bound/stateB/run1/transitions/frame1.gro",
    #         )
    #     )
    #     self.assertGreater(stat_inf.st_size, 2500000)

    #     stat_inf = os.stat(
    #         os.path.join(
    #             self._test_dir,
    #             "0ec09ef_4afa8f9/unbound/stateA/run1/transitions/frame1.gro",
    #         )
    #     )
    #     self.assertGreater(stat_inf.st_size, 357800)

    #     stat_inf = os.stat(
    #         os.path.join(
    #             self._test_dir,
    #             "0ec09ef_4afa8f9/unbound/stateB/run1/transitions/frame1.gro",
    #         )
    #     )
    #     self.assertGreater(stat_inf.st_size, 357800)

    #     step_conf = {
    #         _SBE.STEPID: "prepare_simulations",
    #         _SBE.STEP_TYPE: "pmx_prepare_simulations",
    #         _SBE.EXEC: {
    #             "platform": "slurm",
    #             "resources": {
    #                 "partition": "gpu",
    #                 "gres": "gpu:1",
    #                 "modules": [
    #                     "GROMACS/2021-fosscuda-2019a-PLUMED-2.7.1-Python-3.7.2"
    #                 ],
    #             },
    #         },
    #         _SBE.SETTINGS: {
    #             _SBE.SETTINGS_ARGUMENTS: {
    #                 _SBE.SETTINGS_ARGUMENTS_FLAGS: [],
    #                 _SBE.SETTINGS_ARGUMENTS_PARAMETERS: {},
    #             },
    #             _SBE.SETTINGS_ADDITIONAL: {"sim_type": "transitions"},
    #         },
    #     }

    #     step_run_simulations = StepPMXRunSimulations(**step_conf)
    #     step_run_simulations.work_dir = self._test_dir
    #     step_run_simulations._workflow_object = WorkFlow()
    #     step_run_simulations.get_workflow_object().workflow_data.perturbation_map = (
    #         self.p_map
    #     )
    #     step_run_simulations.execute()

    #     stat_inf = os.stat(
    #         glob(
    #             os.path.join(
    #                 self._test_dir,
    #                 "0ec09ef_4afa8f9/bound/stateB/run1/transitions/*.sh",
    #             )
    #         )[0]
    #     )
    #     self.assertGreater(stat_inf.st_size, 52000)

    #     stat_inf = os.stat(
    #         os.path.join(
    #             self._test_dir, "0ec09ef_4afa8f9/unbound/stateB/run1/transitions/md.log"
    #         )
    #     )

    #     self.assertGreater(stat_inf.st_size, 1339800)

    #     stat_inf = os.stat(
    #         os.path.join(
    #             self._test_dir,
    #             "0cd4b47_4f2ffa1/bound/stateB/run1/transitions/dhdl1.xvg",
    #         )
    #     )
    #     self.assertEqual(stat_inf.st_size, 72216)
