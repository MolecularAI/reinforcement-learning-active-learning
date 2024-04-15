import unittest
import pytest
import os
from icolos.core.workflow_steps.pmx.run_analysis import StepPMXRunAnalysis
from icolos.utils.enums.step_enums import StepBaseEnum
from tests.tests_paths import (
    PATHS_EXAMPLEDATA,
    create_test_dir,
    MAIN_CONFIG,
    export_unit_test_env_vars,
    get_ligands_as_compounds_with_conformers,
)
from icolos.utils.general.files_paths import attach_root_path
from icolos.core.composite_agents.workflow import WorkFlow
from icolos.core.containers.perturbation_map import PerturbationMap

_SBE = StepBaseEnum


# FIXME fails sometimes, might be due to unclean directories
class Test_PMXanalyse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._test_dir = attach_root_path("tests/junk/pmx/analyse")
        #
        create_test_dir(PATHS_EXAMPLEDATA.RUN_ANALYSIS_TEST_DIR, cls._test_dir)

    def setUp(self):
        self.compounds = get_ligands_as_compounds_with_conformers(
            PATHS_EXAMPLEDATA.PMX_TNKS_LIGANDS
        )
        p_map = PerturbationMap(compounds=self.compounds)
        p_map.parse_map_file(file_path=PATHS_EXAMPLEDATA.PMX_TNKS_MAP)
        self.p_map = p_map
        self.p_map.replicas = 1
        export_unit_test_env_vars()

    def test_pmx_analyse(self):
        step_conf = {
            _SBE.STEPID: "prepare_simulations",
            _SBE.STEP_TYPE: "pmx_analyse",
            _SBE.EXEC: {
                _SBE.EXEC_PREFIXEXECUTION: MAIN_CONFIG["PMX"]["MODULE"],
            },
            _SBE.SETTINGS: {
                _SBE.SETTINGS_ARGUMENTS: {
                    _SBE.SETTINGS_ARGUMENTS_FLAGS: ["--quiet"],
                    _SBE.SETTINGS_ARGUMENTS_PARAMETERS: {},
                },
                _SBE.SETTINGS_ADDITIONAL: {},
            },
        }
        step_pmx_analyse = StepPMXRunAnalysis(**step_conf)
        step_pmx_analyse.work_dir = self._test_dir
        step_pmx_analyse._workflow_object = WorkFlow()
        step_pmx_analyse._workflow_object.workflow_data.perturbation_map = self.p_map
        step_pmx_analyse.data.compounds = self.compounds
        step_pmx_analyse.execute()

        stat_inf = os.stat(
            os.path.join(self._test_dir, "0ec09ef_4afa8f9/bound/analyse1/results.txt")
        )

        self.assertGreater(stat_inf.st_size, 18000)

        stat_inf = os.stat(os.path.join(self._test_dir, "resultsAll.csv"))

        self.assertGreater(stat_inf.st_size, 200)

    @pytest.mark.xfail(reason="results.txt can have very different sizes depending on test environment")
    def test_pmx_analyse_experimental_writeout(self):
        step_conf = {
            _SBE.STEPID: "prepare_simulations",
            _SBE.STEP_TYPE: "pmx_analyse",
            _SBE.EXEC: {
                _SBE.EXEC_PREFIXEXECUTION: MAIN_CONFIG["PMX"]["MODULE"],
            },
            _SBE.SETTINGS: {
                _SBE.SETTINGS_ARGUMENTS: {
                    _SBE.SETTINGS_ARGUMENTS_FLAGS: [],
                    _SBE.SETTINGS_ARGUMENTS_PARAMETERS: {},
                },
                _SBE.SETTINGS_ADDITIONAL: {
                    "exp_results": MAIN_CONFIG["PMX"]["EXP_DATA"]
                },
            },
        }
        step_pmx_analyse = StepPMXRunAnalysis(**step_conf)
        step_pmx_analyse.work_dir = self._test_dir
        step_pmx_analyse._workflow_object = WorkFlow()
        step_pmx_analyse._workflow_object.workflow_data.perturbation_map = self.p_map
        step_pmx_analyse.data.compounds = self.compounds
        step_pmx_analyse.execute()

        stat_inf = os.stat(
            os.path.join(self._test_dir, "0ec09ef_4afa8f9/bound/analyse1/results.txt")
        )

        # FIXME file size can vary due to length of paths
        self.assertGreater(stat_inf.st_size, 18000)

        stat_inf = os.stat(os.path.join(self._test_dir, "resultsAll.csv"))

        self.assertGreater(stat_inf.st_size, 200)

        stat_inf = os.stat(os.path.join(self._test_dir, "resultsSummary.csv"))

        self.assertGreater(stat_inf.st_size, 130)

        self.assertEqual(
            step_pmx_analyse.data.compounds[8]
            .get_enumerations()[0]
            .get_conformers()[0]
            .get_molecule()
            .GetProp("ddG"),
            "11.21",
        )
