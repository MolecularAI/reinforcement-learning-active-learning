from icolos.core.containers.generic import GenericData
import unittest
import os
from icolos.core.containers.gmx_state import GromacsState
from icolos.utils.enums.step_enums import StepBaseEnum, StepGromacsEnum
from tests.tests_paths import MAIN_CONFIG, PATHS_EXAMPLEDATA, export_unit_test_env_vars
from icolos.core.workflow_steps.gromacs.grompp import StepGMXGrompp
from icolos.utils.general.files_paths import attach_root_path

_SGE = StepGromacsEnum()
_SBE = StepBaseEnum


class Test_Grompp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._test_dir = attach_root_path("tests/junk/gromacs")
        if not os.path.isdir(cls._test_dir):
            os.makedirs(cls._test_dir)

        export_unit_test_env_vars()

    def setUp(self):
        with open(
            attach_root_path(PATHS_EXAMPLEDATA.GROMACS_HOLO_STRUCTURE_GRO), "r"
        ) as f:
            struct = f.readlines()
        with open(PATHS_EXAMPLEDATA.GROMACS_IONS_MDP, "r") as f:
            self.mdp = f.read()

        self.topol = GromacsState()
        self.topol.structures = [GenericData(_SGE.STD_STRUCTURE, file_data=struct)]
        top_path = os.path.dirname(PATHS_EXAMPLEDATA.GROMACS_1BVG_TOP)
        top_file = PATHS_EXAMPLEDATA.GROMACS_1BVG_TOP.split("/")[-1]
        self.topol.parse(top_path, top_file)

    def test_grompp(self):
        step_conf = {
            _SBE.STEPID: "test_grompp",
            _SBE.STEP_TYPE: "grompp",
            _SBE.EXEC: {
                _SBE.EXEC_PREFIXEXECUTION: "module load GROMACS/2021-fosscuda-2019a-PLUMED-2.7.1-Python-3.7.2"
            },
            _SBE.SETTINGS: {
                _SBE.SETTINGS_ARGUMENTS: {
                    _SBE.SETTINGS_ARGUMENTS_FLAGS: [],
                    _SBE.SETTINGS_ARGUMENTS_PARAMETERS: {},
                },
                _SBE.SETTINGS_ADDITIONAL: {
                    _SGE.FIELDS: {
                        "nsteps": 50,
                        "-nsteeps": 123,
                    },  # deliberate typo to check warning
                    _SGE.FORCEFIELD: MAIN_CONFIG["FORCEFIELD"],
                    "restraints": False,
                    _SGE.MAKE_NDX_COMMAND: "auto",
                },
            },
        }

        step_grompp = StepGMXGrompp(**step_conf)

        step_grompp.data.generic.add_file(
            GenericData(file_name="tmp03394.mdp", file_data=self.mdp, argument=True)
        )
        step_grompp.data.gmx_state = self.topol
        step_grompp.execute()

        out_path = os.path.join(self._test_dir, _SGE.STD_TPR)
        step_grompp.get_topol().write_tpr(self._test_dir)
        stat_inf = os.stat(out_path)
        self.assertEqual(stat_inf.st_size, 1528248)
