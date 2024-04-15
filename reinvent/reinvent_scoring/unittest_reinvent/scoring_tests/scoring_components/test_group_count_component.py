import unittest

import numpy.testing as npt
from reinvent_chemistry import Conversions

from reinvent_scoring import TransformationParametersEnum
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums.transformation_type_enum import TransformationTypeEnum
from reinvent_scoring.scoring.score_components.standard.group_count import GroupCount
from unittest_reinvent.fixtures.test_data import CELECOXIB


class TestGroupCount(unittest.TestCase):

    def setUp(self):
        self._chemistry = Conversions()
        transformation_type = TransformationTypeEnum()
        component_specific_parameters = ComponentSpecificParametersEnum()
        sf_enum = ScoringFunctionComponentNameEnum()
        csp_enum = ComponentSpecificParametersEnum()
        specific_parameters = {
            TransformationParametersEnum.TRANSFORMATION_TYPE: transformation_type.NO_TRANSFORMATION,
            component_specific_parameters.SMILES: "[F,Cl]",
            csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.HIGH: 3,
                TransformationParametersEnum.LOW: 1,
                TransformationParametersEnum.TRANSFORMATION_TYPE: transformation_type.STEP
            }
        }
        parameters = ComponentParameters(component_type=sf_enum.GROUP_COUNT,
                                         name="group count",
                                         weight=1.,
                                         specific_parameters=specific_parameters)
        self.component = GroupCount(parameters)
        self.smile = CELECOXIB
        self.molecule = self._chemistry.smile_to_mol(self.smile)

    def test_one_molecule(self):
        score = self.component.calculate_score([self.molecule])
        self.assertEqual(1, len(score.total_score))
        npt.assert_equal(score.total_score[0], 1)

    def test_two_molecules(self):
        score = self.component.calculate_score([self.molecule, self.molecule])
        self.assertEqual(2, len(score.total_score))
        npt.assert_equal(score.total_score[0], 1)
        npt.assert_equal(score.total_score[1], 1)
