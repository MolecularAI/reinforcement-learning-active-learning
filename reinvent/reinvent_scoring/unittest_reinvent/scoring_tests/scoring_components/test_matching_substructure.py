import numpy.testing as npt

from reinvent_scoring.scoring.score_components import MatchingSubstructure
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single
from unittest_reinvent.fixtures.test_data import COCAINE, CAFFEINE, CELECOXIB
from unittest_reinvent.scoring_tests.scoring_components.base_matching_substructure import \
    BaseTestMatchingSubstructure


class TestMatchingSubstructures(BaseTestMatchingSubstructure):

    def setUp(self):
        self.smiles = [COCAINE]
        super().setUp()
        self.component = MatchingSubstructure(self.parameters)

    def test_match_1(self):
        npt.assert_almost_equal(score_single(self.component, CAFFEINE), 0.5)

    def test_match_2(self):
        npt.assert_almost_equal(score_single(self.component, CELECOXIB), 0.5)

    def test_not_use_chirality(self):
        self.parameters.specific_parameters["smiles"] = ['C[C@H](F)Cl']
        self.component = MatchingSubstructure(self.parameters)
        npt.assert_almost_equal(score_single(self.component, 'CC[C@H](F)Cl'), 1)

        self.parameters.specific_parameters["smiles"] = ['C[C@@H](F)Cl']
        self.component = MatchingSubstructure(self.parameters)
        npt.assert_almost_equal(score_single(self.component, 'CC[C@H](F)Cl'), 1)

        self.parameters.specific_parameters["smiles"] = ['CC(F)Cl']
        self.component = MatchingSubstructure(self.parameters)
        npt.assert_almost_equal(score_single(self.component, 'CC[C@H](F)Cl'), 1)

    def test_use_chirality(self):
        self.parameters.specific_parameters["use_chirality"] = True

        self.parameters.specific_parameters["smiles"] = ['C[C@H](F)Cl']
        self.component = MatchingSubstructure(self.parameters)
        npt.assert_almost_equal(score_single(self.component, 'CC[C@H](F)Cl'), 1)

        self.parameters.specific_parameters["smiles"] = ['C[C@@H](F)Cl']
        self.component = MatchingSubstructure(self.parameters)
        npt.assert_almost_equal(score_single(self.component, 'CC[C@H](F)Cl'), 0.5)

        self.parameters.specific_parameters["smiles"] = ['CC(F)Cl']
        self.component = MatchingSubstructure(self.parameters)
        npt.assert_almost_equal(score_single(self.component, 'CC[C@H](F)Cl'), 1)



