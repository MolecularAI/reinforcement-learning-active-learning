import unittest
import numpy.testing as npt
import pytest

from reinvent_chemistry.conversions import Conversions

from reinvent_scoring.scoring.score_components.console_invoked.mmp.mmp import MMP
from reinvent_scoring.scoring.score_components.console_invoked.mmp.mmp_parameter_dto import MMPParameterDTO
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_mmp_component_parameters
from unittest_reinvent.fixtures.test_data import CHEMBL4442703, CHEMBL4440201, CHEMBL4475970, CHEMBL4530872, \
    CHEMBL4475647, CHEMBL4458766, CHEMBL4546342, CHEMBL4469449
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum


@pytest.mark.integration
class TestMMP(unittest.TestCase):
    def setUp(self):
        self.chemistry = Conversions()
        self.specific_parameters_enums = ComponentSpecificParametersEnum()
        self.parameters = create_mmp_component_parameters()
        self.query_smiles = [CHEMBL4442703, CHEMBL4469449, CHEMBL4546342, CHEMBL4458766, CHEMBL4530872, CHEMBL4475647]
        self.query_mols = [self.chemistry.smile_to_mol(smile) for smile in self.query_smiles]

    def _init_with_single_reference_molecule(self):
        self.mmp_dto = MMPParameterDTO(mmp_reference_molecules=[CHEMBL4440201])
        self.parameters.specific_parameters.update(vars(self.mmp_dto))
        self.component = MMP(self.parameters)

    def _init_with_multiple_reference_molecule(self):
        self.mmp_dto = MMPParameterDTO( mmp_reference_molecules=[CHEMBL4440201, CHEMBL4475970])
        self.parameters.specific_parameters.update(vars(self.mmp_dto))
        self.component = MMP(self.parameters)

    def test_MMP_single_reference_molecule(self):
        self._init_with_single_reference_molecule()
        summary = self.component.calculate_score(self.query_mols)
        npt.assert_almost_equal(summary.total_score, [1,1,1,0.5,0.5,0.5])

    def test_MMP_multiple_reference_molecule(self):
        self._init_with_multiple_reference_molecule()
        summary = self.component.calculate_score(self.query_mols)
        npt.assert_almost_equal(summary.total_score, [1,1,1,0.5,1,1])
