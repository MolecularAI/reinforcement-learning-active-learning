import pytest

from rdkit import Chem

from reinvent_scoring.scoring.score_components.standard.chemprop_component import (
    ChemPropComponent,
)
from reinvent_scoring.scoring.component_parameters import ComponentParameters


@pytest.mark.integration
def test_chemprop():
    cp = ComponentParameters(
        "",
        "",
        1.0,
        {
            "checkpoint_dir": "../fixtures/chemprop_models",
            "rdkit_2d_normalized": True,
            "transformation": {
                "transformation_type": "reverse_sigmoid",
                "high": -5.0,
                "low": -35.0,
                "k": 0.2,
            },
        },
    )

    component = ChemPropComponent(cp)

    smilies = [
        "Cc1cc(cc(c1O)C)CNC(=O)CSc2ccc(cn2)S(=O)(=O)N3CC[NH+](CC3)Cc4ccccc4",  # -29.21
        "c1ccc-2c(c1)Cc3c2cc(cc3)NC(=O)c4cc(ccc4Br)F",  # -21.10
        "Cn1cc(c(n1)C(F)(F)F)S(=O)(=O)N",  # -13.35
        "CN1C[C@](SC1=S)([C@H]([C@@H]([C@@H](CO)O)O)O)O",  # -8.63
        "INVALID",  # 0.0
        "Cc1c(cn(n1)C)C(=O)N2[C@@H]3CCCC[C@@H]3C[C@H]2C(=O)[O-]",
    ]  # -11.14
    mols = [Chem.MolFromSmiles(smiles) for smiles in smilies]
    score_summary = component.calculate_score(mols)

    for score in score_summary.total_score:
      assert 0.0 < score < 1.0
