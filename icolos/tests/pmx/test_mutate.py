import shutil
import unittest
import pytest
import os
from icolos.core.composite_agents.workflow import WorkFlow
from icolos.core.containers.generic import GenericData
from icolos.core.workflow_steps.pmx.gentop import StepPMXgentop
from icolos.core.workflow_steps.pmx.mutate import StepPMXmutate
from icolos.utils.enums.step_enums import StepBaseEnum, StepGromacsEnum
from tests.tests_paths import MAIN_CONFIG, PATHS_EXAMPLEDATA, export_unit_test_env_vars
from icolos.utils.general.files_paths import attach_root_path


_SBE = StepBaseEnum
_SGE = StepGromacsEnum()


class Test_PMXmutate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._test_dir = attach_root_path("tests/junk/pmx/mutate")
        if not os.path.isdir(cls._test_dir):
            os.makedirs(cls._test_dir)
        if os.path.exists(cls._test_dir):
            shutil.rmtree(cls._test_dir)

        export_unit_test_env_vars()

    def setUp(self):
        with open(PATHS_EXAMPLEDATA.PMX_MUTATIONS_PROTEIN, "r") as f:
            data = f.read()
        self.system = GenericData(file_name="protein.pdb", file_data=data)
        with open(PATHS_EXAMPLEDATA.PMX_MUTATIONS_LIST, "r") as f:
            muts = f.read()
        self.muts = GenericData(file_name="mutations.mut", file_data=muts)
        with open(PATHS_EXAMPLEDATA.PMX_GENTOP_TOPOLOGY, "r") as f:
            top = f.read()
        self.top = GenericData(file_name="topol.top", file_data=top)

    @pytest.mark.xfail(reason="mdp files missing from IcolosData")
    def test_pmx_mutate(self):
        mutate_conf = {
            _SBE.STEPID: "01_PMX_MUTATE",
            _SBE.STEP_TYPE: _SBE.STEP_PMX_MUTATE,
            _SBE.EXEC: {
                _SBE.EXEC_PREFIXEXECUTION: MAIN_CONFIG["PMX"]["MODULE"],
                _SBE.EXEC_PARALLELIZATION: {
                    _SBE.EXEC_PARALLELIZATION_CORES: 8,
                    _SBE.EXEC_PARALLELIZATION_MAXLENSUBLIST: 1,
                },
            },
            _SBE.SETTINGS: {
                _SBE.SETTINGS_ADDITIONAL: {
                    # settings for protein parametrisation
                    "forcefield": "amber14sbmut",
                    "water": "tip3p",
                },
            },
        }

        step_mutate = StepPMXmutate(**mutate_conf)
        step_mutate.data.generic.add_file(self.muts)
        step_mutate.data.generic.add_file(self.system)

        step_mutate.work_dir = self._test_dir
        step_mutate._workflow_object = WorkFlow()
        step_mutate.execute()
        stat_inf = os.stat(os.path.join(self._test_dir, "wt_P79ASP/bound/topol.top"))
        self.assertGreater(stat_inf.st_size, 1500)

        stat_inf = os.stat(os.path.join(self._test_dir, "wt_P79ASP/bound/init.pdb"))
        self.assertGreater(stat_inf.st_size, 2800)

        stat_inf = os.stat(os.path.join(self._test_dir, "wt_P77TYR/bound/topol.top"))
        self.assertGreater(stat_inf.st_size, 1500)

        stat_inf = os.stat(os.path.join(self._test_dir, "wt_P77TYR/bound/init.pdb"))
        self.assertGreater(stat_inf.st_size, 2800)
        stat_inf = os.stat(os.path.join(self._test_dir, "wt_P76GLY/bound/topol.top"))
        self.assertGreater(stat_inf.st_size, 1500)

        stat_inf = os.stat(os.path.join(self._test_dir, "wt_P76GLY/bound/init.pdb"))
        self.assertGreater(stat_inf.st_size, 2800)

        stat_inf = os.stat(os.path.join(self._test_dir, "wt_P76GLY/unbound/init.pdb"))
        self.assertGreater(stat_inf.st_size, 2800)

        gentop_conf = {
            _SBE.STEPID: "01_PMX_GENTOP",
            _SBE.STEP_TYPE: _SBE.STEP_PMX_MUTATE,
            _SBE.EXEC: {
                _SBE.EXEC_PREFIXEXECUTION: MAIN_CONFIG["PMX"]["MODULE"],
                _SBE.EXEC_PARALLELIZATION: {
                    _SBE.EXEC_PARALLELIZATION_CORES: 8,
                    _SBE.EXEC_PARALLELIZATION_MAXLENSUBLIST: 1,
                },
            },
            _SBE.SETTINGS: {
                _SBE.SETTINGS_ADDITIONAL: {},
            },
        }

        step_gentop = StepPMXgentop(**gentop_conf)
        step_gentop.data.generic.add_file(self.top)

        step_gentop.work_dir = self._test_dir
        wf = step_mutate.get_workflow_object()

        step_gentop.set_workflow_object(wf)

        step_gentop.execute()
        stat_inf = os.stat(os.path.join(self._test_dir, "wt_P76GLY/bound/pmxtop.top"))
        self.assertGreater(stat_inf.st_size, 300)
        stat_inf = os.stat(os.path.join(self._test_dir, "wt_P76GLY/unbound/pmxtop.top"))
        self.assertGreater(stat_inf.st_size, 300)
