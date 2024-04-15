from reinvent_scoring import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.component_parameters import ComponentParameters

from reinvent_scoring.scoring.score_components.aizynth.building_block_availability_component import (
    drop_atommap,
)


def test_drop_atommap():
    assert drop_atommap("CCC") == "CCC"
    assert drop_atommap("CCC[C:0]C[C:12]") == "[C]C[C]CCC"
    assert drop_atommap("[C:0]CC([O:1])=[O:0]") == "[C]CC([O])=O"
