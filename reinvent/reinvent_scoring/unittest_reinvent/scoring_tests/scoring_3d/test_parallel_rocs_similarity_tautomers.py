import unittest
import pytest
import numpy.testing as npt

from reinvent_scoring.scoring import CustomSum
from unittest_reinvent.fixtures.test_config import ROCS_SIMILARITY_TEST_DATA
from reinvent_scoring.scoring.enums import ROCSSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum

from unittest_reinvent.fixtures.test_data import AMOXAPINE, METHOXYHYDRAZINE, CAFFEINE, PARACETAMOL
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestParallelRocsSimilarityTautomers(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        rsp_enum = ROCSSpecificParametersEnum()
        specific_parameters = {rsp_enum.SHAPE_WEIGHT: 0.5,
                               rsp_enum.COLOR_WEIGHT: 0.5,
                               rsp_enum.ROCS_INPUT: ROCS_SIMILARITY_TEST_DATA,
                               rsp_enum.MAX_CPUS: 4
                               }
        ts_parameters = component_parameters(component_type=sf_enum.PARALLEL_ROCS_SIMILARITY,
                                             name="parallel_rocs_similarity",
                                             specific_parameters=specific_parameters)
        self.sf_state = CustomSum(parameters=[ts_parameters])

    def test_parallel_rocs_similarity_1(self):
        smiles = [AMOXAPINE, PARACETAMOL, METHOXYHYDRAZINE, CAFFEINE]
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, [0.41, 0.35, 0.1 , 0.34], 2)
