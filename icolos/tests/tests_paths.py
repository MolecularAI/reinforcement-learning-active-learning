import unittest
import pytest
from icolos.utils.enums.program_parameters import PantherEnum
from icolos.core.containers.generic import GenericData
import json
import os
from typing import List, Dict
from icolos.core.containers.compound import Compound, Enumeration, Conformer
from icolos.utils.general.files_paths import attach_root_path
from icolos.utils.smiles import to_smiles
from rdkit import Chem
from icolos.utils.enums.write_out_enums import WriteOutEnum
from shutil import copytree, rmtree, which

_PE = PantherEnum()
_WE = WriteOutEnum()

# load the instantiated "config.json", holding the license key for OpenEye for example
try:
    with open(
        attach_root_path("src/icolos/config/unit_tests_config/config.json"), "r"
    ) as f:
        MAIN_CONFIG = json.load(f)
except Exception as e:
    print(e)
    MAIN_CONFIG = {}


def has_slurm() -> bool:
    return which("sbatch") is not None


skip_without_slurm = pytest.mark.skipif(condition=not has_slurm(), reason="Not running on a slurm system")


def expand_path(path: str) -> str:
    return os.path.join(MAIN_CONFIG["ICOLOS_TEST_DATA"], path)


def create_test_dir(source: str, dest: str) -> None:
    try:
        if os.path.isdir(dest):
            # remove the existing directory structure before calling copytree or it will complain
            rmtree(dest)
        copytree(source, dest)
    except Exception as e:
        os.makedirs(dest, exist_ok=True)


def export_unit_test_env_vars():
    # make sure "PATH" is executed last to expand upwards variables
    for key in MAIN_CONFIG.keys():
        if key != "PATH":
            if isinstance(MAIN_CONFIG[key], str):
                os.environ[str(key)] = os.path.expandvars(MAIN_CONFIG[key])
            # iterate through nested dicts
            elif isinstance(MAIN_CONFIG[key], dict):
                for k in MAIN_CONFIG[key].keys():
                    os.environ[str(k)] = os.path.expandvars(MAIN_CONFIG[key][k])
    if "PATH" in MAIN_CONFIG.keys():
        os.environ["PATH"] = os.path.expandvars(MAIN_CONFIG["PATH"])


class FloatComparisonMixin:
    def assertListAlmostEqual(self, a, b, places=3):
        return all(self.assertAlmostEqual(aa, bb, places=places) for aa, bb in zip(a, b))


class PATHS_1UYD:

    GRID_PATH = expand_path("Glide/1UYD_grid_no_constraints.zip")
    GRID_CONSTRAINTS_PATH = expand_path("Glide/1UYD_grid_constraints.zip")
    PDBQT_PATH = expand_path("AutoDockVina/1UYD_fixed.pdbqt")
    PDB_PATH = expand_path("molecules/1UYD/1UYD_apo.pdb")
    HOLO_PDB = expand_path("molecules/1UYD/1UYD_holo.pdb")
    APO_MAE = expand_path("molecules/1UYD/1UYD_apo.mae")
    LIGANDS = expand_path("molecules/1UYD/1UYD_ligands.sdf")
    GOLD_MOL2_PROTEIN = "Gold/1UYD_protein.mol2"
    GOLD_CAVITY_MOL2 = "molecules/1UYD/PU8_reference_ligand.mol2"
    NATIVE_LIGAND_SDF = expand_path("molecules/1UYD/PU8_native_ligand.sdf")
    NATIVE_LIGAND_PDB = expand_path("molecules/1UYD/PU8_native_ligand.pdb")
    LIG4_POSES = expand_path("fep_plus/1UYD_ligand_subset.sdf")
    XRAY_STRUCTURES = expand_path("fep_plus/xray_structures")

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")


class PATHS_EXAMPLEDATA:

    ASPIRIN_SMI_PATH = expand_path("molecules/aspirin.smi")
    PARACETAMOL_SMI_PATH = expand_path("molecules/paracetamol.smi")
    ASPIRIN_PATH = expand_path("molecules/aspirin.sdf")
    PARACETAMOL_PATH = expand_path("molecules/paracetamol.sdf")
    SMALL_MOLECULES_SMI_PATH = expand_path("molecules/small_molecules.smi")
    SMALL_MOLECULES_CSV_PATH = expand_path("molecules/small_molecules.csv")
    SMALL_MOLECULES_CSV_PATH_DELIMITER_SEMICOLON = expand_path(
        "molecules/small_molecules_semicolon.csv"
    )
    MEDIUM_MOLECULES_SMI_PATH = expand_path("molecules/medium_molecules.smi")
    SMALL_MOLECULES_SDF_PATH = expand_path("molecules/small_molecules.sdf")
    SMALL_MOLECULES_JSON_PATH = expand_path("reinvent/small_input.json")
    MEDIUM_MOLECULES_SDF_PATH = expand_path("molecules/medium_molecules.sdf")
    PARACETAMOL_MULTIPLE_CONF = expand_path("molecules/paracetamol_multiple.sdf")
    PARACETAMOL_COSMO = expand_path("Turbomole/paracetamol.cosmo")
    PARACETAMOL_COSMO_OUTPUT = expand_path("cosmo/cosmotherm.out")
    EPSA_MODEL_PATH = expand_path("models/ePSA_example.pkl")
    EPSA_BOLTZMANN_WEIGHTING_EXAMPLE_MOLECULE = expand_path(
        "models/ePSA_Boltzmann_weighting.sdf"
    )
    VECTOR_BOLTZMANN_WEIGHTING_EXAMPLE_MOLECULE = expand_path(
        "molecules/turbomol_nmr.sdf"
    )
    GLIDE_EXAMPLE_IN = expand_path("Glide/example.in")
    GOLD_EXAMPLE_CONFIG = "Gold/gold.conf"
    EPSA_EXAMPLE_MOLECULE = expand_path("models/ePSA_example_mol.sdf")
    PRIME_RECEPTOR_COX2 = expand_path("prime/cox2_receptor.mae")
    PRIME_COX2_GRID = expand_path("molecules/1CX2/1cx2_GridGen.zip")
    PRIME_DOCKED_LIGAND_SDF = expand_path("prime/docked_ligand.sdf")
    CLUSTERING_11CONFS = expand_path("clustering/paracetamol_11_conformers.sdf")
    PANTHER_CONFIG = expand_path("panther/default_panther.in")
    PANTHER_NEGATIVE_IMAGE = expand_path("panther/1uyd_negative_image.mol2")
    PANTHER_HOLO_PDB = expand_path("panther/1UYD_holo_residue_X.pdb")
    SHAEP_LIGAND_DOCKED_POSE = expand_path("panther/cox2_ligand_bound.sdf")

    GROMACS_NVT_MDP = expand_path("gromacs/protein/nvt_equil.mdp")
    GROMACS_NPT_MDP = expand_path("gromacs/protein/npt_equil.mdp")
    GROMACS_MINIM_MDP = expand_path("gromacs/protein/minim.mdp")
    GROMACS_IONS_MDP = expand_path("gromacs/protein/ions.mdp")
    GROMACS_MD_MDP = expand_path("gromacs/protein/md.mdp")
    GROMACS_HOLO_STRUCTURE = expand_path("gromacs/protein/1BVG.pdb")
    GROMACS_DNA_STRUCTURE = expand_path("gromacs/cccc.pdb")
    GROMACS_HOLO_STRUCTURE_GRO = expand_path("gromacs/protein/1BVG.gro")
    GROMACS_1BVG_INDEX = expand_path("gromacs/protein/index.ndx")
    GROMACS_DMP_LIGAND_TRJ = expand_path("gromacs/protein/DMP.xtc")
    GROMACS_DMP_LIGAND_SDF = expand_path("gromacs/protein/DMP.sdf")
    GROMACS_PROTEIN_FILE_BASE = expand_path("gromacs/protein")
    GROMACS_TS_CLUSTERS = expand_path("gromacs/protein/cluster_id.xvg")
    GROMACS_1BVG_TPR = expand_path("gromacs/protein/1BVG.tpr")
    GROMACS_1BVG_XTC = expand_path("gromacs/protein/1BVG.xtc")
    GROMACS_1BVG_TOP = expand_path("gromacs/protein/1BVG.top")
    GROMACS_1BVG_PDB = expand_path("gromacs/protein/1BVG.pdb")
    MMPBSA_CUSTOM_INPUT = expand_path("gromacs/test_input_mmpbsa.in")
    MMPBSA_POSRE = expand_path("gromacs/protein/posre.itp")
    MMPBSA_LIG_POSRE = expand_path("gromacs/protein/posre_DMP:100.itp")
    MMPBSA_LIG_ITP = expand_path("gromacs/protein/DMP:100.itp")

    FEP_PLUS_DOCKING_PV = expand_path("fep_plus/1UYD_ligands_pv.maegz")
    FEP_PLUS_EXAMPLE_FMP = expand_path("fep_plus/out.fmp")
    FEP_PLUS_MAP_LOG = expand_path("fep_plus/fep_mapper.log")
    FEP_PLUS_MAP_LOG_MIN = expand_path("fep_plus/fep_mapper_min.log")
    FEP_PLUS_MAP_LOG_SINGLE_EDGE = expand_path("fep_plus/fep_mapper_single_edge.log")

    FEP_PLUS_LIGANDS = expand_path("fep_plus/ligprep_confs.sdf")
    FEP_PLUS_EXAMPLE_FMP_OUT = expand_path("fep_plus/test_out.fmp")
    FEP_PLUS_MULTISIM = expand_path("fep_plus/multisim.log")
    FEP_PLUS_PROTEIN = expand_path("fep_plus/<FILE>.pdb")
    FEP_PLUS_OTHER_PROTEIN = expand_path("fep_plus/<FILE>_apo.pdb")
    FEP_PLUS_MULTISIM_LONG = expand_path("fep_plus/multisim.log")

    MODEL_BUILDER_EXAMPLE_JSON = expand_path(
        "model_building/OptunaAZ_example_config.json"
    )
    MODEL_BUILDER_TEST_INPUT_SDF = expand_path("model_building/test_input_data.sdf")
    COX2_ACTIVES_DOCKED = expand_path("molecules/1CX2/docked_actives.sdf")

    CAVITY_TRJ_FOLDER = expand_path("fpocket/parch_align_trj")
    CAVITY_DTR_FILE = expand_path("fpocket/parch_align_trj/clickme.dtr")
    CAVITY_CMS_FILE = expand_path("fpocket/parch_align_trj/out.cms")
    MDPOCKET_XTC_FILE = expand_path("fpocket/1BVG_apo.xtc")
    MDPOCKET_PDB_FILE = expand_path("fpocket/1BVG_apo.pdb")
    MDPOCKET_PDB_FILE_DRY = expand_path("fpocket/structure_0.pdb")
    MD_POCKET_DESMOND_TOP = expand_path("fpocket/top.pdb")

    DESMOND_SETUP_PDB = expand_path("desmond/1cx2.pdb")
    DESMOND_PRODUCTION_CMS = expand_path("desmond/setup.cms")
    TEST_FASTA_FILE = expand_path("molecules/peptides/1acw.fasta")

    LIGAND_HYBRID_TEST_DIR = expand_path("pmx/ligand_hybrid")
    PREPARE_SIMULATIONS_TEST_DIR = expand_path("pmx/prepare_simulations")
    ATOM_MAPPING_TEST_DIR = expand_path("pmx/atom_mapping")
    ASSEMBLE_SYSTEMS_TEST_DIR = expand_path("pmx/assemble_systems")
    BOX_WATER_IONS_TEST_DIR = expand_path("pmx/box_water_ions")
    PREPARE_TRANSITIONS_TEST_DIR = expand_path("pmx/prepare_transitions")
    RUN_ANALYSIS_TEST_DIR = expand_path("pmx/analyse")
    PMX_MDP_FILES = expand_path("pmx/assemble_systems/input/mdp")
    PMX_MUTATIONS_LIST = expand_path("pmx/mutate/mutations.mut")
    PMX_MUTATIONS_PROTEIN = expand_path("pmx/mutate/2flu.pdb")
    PROTIEN_INTERACTION_9MER = expand_path("pmx/mutate/2flu_p.sdf")
    PMX_GENTOP_TOPOLOGY = expand_path("pmx/gentop/topol.top")

    RUN_SIMULATIONS_TEST_DIR = expand_path("pmx/run_simulations")
    PMX_TNKS_TEST_DIR = expand_path("pmx")
    PMX_TNKS_MAP = expand_path("pmx/fep_mapper.log")
    PMX_TNKS_LIGANDS = expand_path("pmx/ligands.sdf")
    PMX_TNKS_PROTEIN = expand_path("pmx/4ui5.pdb")
    PMX_ABFE_PROTEIN = expand_path("pmx/5tbm.pdb")
    DSSP_PDB_1 = expand_path("structure_prediction/1e0n.pdb")
    DSSP_PDB_2 = expand_path("structure_prediction/1jbf.pdb")
    DSSP_PDB_3 = expand_path("structure_prediction/6nox.pdb")

    # try to find the internal value and return

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")


def get_mol_as_Compound(abs_path: str, compound_number: int = 0) -> Compound:
    mol_supplier = Chem.SDMolSupplier(abs_path, removeHs=False)
    for mol in mol_supplier:
        enum = Enumeration(
            smile=to_smiles(mol), original_smile=to_smiles(mol), molecule=mol
        )
        comp = Compound(
            name=os.path.basename(abs_path), compound_number=compound_number
        )
        comp.add_enumeration(enum, auto_update=True)
        return comp


def get_1UYD_ligands_as_Compounds(abs_path: str) -> List[Compound]:
    comp_list = []
    mol_supplier = Chem.SDMolSupplier(abs_path, removeHs=False)
    for cur_id, mol in enumerate(mol_supplier):
        enum = Enumeration(
            smile=to_smiles(mol), original_smile=to_smiles(mol), molecule=mol
        )
        comp = Compound(name=mol.GetProp("_Name"), compound_number=cur_id)
        comp.add_enumeration(enum, auto_update=True)
        comp_list.append(comp)
    return comp_list


def construct_full_compound_object(source) -> List[Compound]:
    def _get_existing_enumeration(comp_id, enum_id):
        comp = _get_existing_compound(comp_id)
        for enum in comp.get_enumerations():
            if enum.get_enumeration_id() == int(enum_id):
                return enum
        raise ValueError

    def _get_existing_compound(idx):
        for comp in list_compounds:
            if int(idx) == comp.get_compound_number():
                return comp
        raise ValueError

    list_compounds = []
    for mol in Chem.SDMolSupplier(source, removeHs=False):
        new_compound = False
        new_enumeration = False
        mol_name = mol.GetProp(_WE.RDKIT_NAME)
        # assuming the mol name follows Icolos conventions
        try:
            id_parts = mol_name.split(":")
            comp_id = id_parts[0]
            enum_id = id_parts[1]

        except:
            comp_id = mol_name
            enum_id = 0
        try:
            # try to find an existing compound with the correct name
            compound = _get_existing_compound(idx=comp_id)
        except ValueError:
            # the compound does not yet exist, create the object
            new_compound = True
            compound = Compound(name=comp_id, compound_number=comp_id)
        try:
            # check whether the enumeration exists
            enumeration = _get_existing_enumeration(comp_id, enum_id)
        except ValueError:
            new_enumeration = True
            enumeration = Enumeration(
                smile=to_smiles(mol), molecule=mol, original_smile=to_smiles(mol)
            )

        if len(id_parts) == 3 and id_parts[2] == "0":
            # i.e. 0:0:0, we have a conformer
            conf = Conformer(
                conformer=mol,
                enumeration_object=enumeration,
                conformer_id=int(id_parts[2]),
            )
            enumeration.add_conformer(conf, auto_update=True)

        if new_enumeration:
            compound.add_enumeration(enumeration, auto_update=True)
        if new_compound:
            list_compounds.append(compound)
    return list_compounds


def get_ligands_as_compounds_with_conformers(
    abs_path: str, poseviewer=None
) -> List[Compound]:
    comp_list = []
    mol_supplier = Chem.SDMolSupplier(abs_path, removeHs=False)
    for cur_id, mol in enumerate(mol_supplier):

        #
        enum = Enumeration(
            smile=to_smiles(mol), original_smile=to_smiles(mol), molecule=mol
        )
        conf = Conformer(conformer=mol)
        conf.get_molecule().SetProp("docking_score", "-12.434")
        if poseviewer is not None:
            conf.add_extra_data("structures_pv.maegz", data=poseviewer)
        enum.add_conformer(conf)
        comp = Compound(name=mol.GetProp("_Name"), compound_number=cur_id)
        comp.add_enumeration(enum, auto_update=True)
        comp_list.append(comp)
    return comp_list


def get_docked_ligands_as_conformers(abs_path: str, poseviewer=None) -> List[Compound]:
    mol_supplier = Chem.SDMolSupplier(abs_path, removeHs=False)
    comp = Compound(name="test_poses", compound_number=1)
    enum = Enumeration()
    for cur_id, mol in enumerate(mol_supplier):
        conf = Conformer(conformer=mol, conformer_id=cur_id)

        if cur_id == 0 and poseviewer is not None:
            conf.add_extra_data(key="structures_pv.maegz", data=poseviewer)

        enum.add_conformer(conf)

    comp.add_enumeration(enum)
    return [comp]


def get_mol_as_Conformer(abs_path: str) -> List[Conformer]:
    list_return = []
    mol_supplier = Chem.SDMolSupplier(abs_path, removeHs=False)
    for mol in mol_supplier:
        list_return.append(Conformer(conformer=mol))
    return list_return


def get_test_Compounds_without_molecules(
    compound_numbers: List[int] = [0],
) -> Dict[str, Compound]:
    """These compounds have neither a molecule in the enumeration nor any Conformer, i.e. no 3D structure."""
    aspirin = Compound(name="Aspirin", compound_number=compound_numbers[0])
    aspirin.add_enumeration(
        Enumeration(
            compound_object=aspirin,
            smile="O=C(C)Oc1ccccc1C(=O)O",
            original_smile="O=C(C)Oc1ccccc1C(=O)O",
        )
    )
    return {"Aspirin": aspirin}


def load_SDF_docked(abs_path: int) -> List[Compound]:
    compounds = []
    mol_supplier = Chem.SDMolSupplier(abs_path, removeHs=False)
    for mol_id, mol in enumerate(mol_supplier):
        comp = Compound(compound_number=mol_id)
        enum = Enumeration(
            smile=str(mol.GetProp("smiles")),
            original_smile=str(mol.GetProp("original_smiles")),
        )
        conf = Conformer(conformer=mol)
        enum.add_conformer(conf, auto_update=True)
        comp.add_enumeration(enum, auto_update=True)
        compounds.append(comp)
    return compounds


def directory_to_generic(path: str) -> List[GenericData]:
    """converts all files in a given path to generic data and returns a list with them"""
    generic_files = []

    for r, d, f in os.walk(path):
        for file in f:
            try:
                with open(os.path.join(r, file), "r") as read_file:
                    data = read_file.read()
                    file_name = file.split("/")[-1]
                    generic_file = GenericData(file_name=file_name, file_data=data)
                    generic_files.append(generic_file)
            except UnicodeDecodeError:
                with open(os.path.join(r, file), "rb") as read_file:
                    data = read_file.read()
                    file_name = file.split("/")[-1]
                    generic_file = GenericData(file_name=file_name, file_data=data)
                    generic_files.append(generic_file)
    return generic_files
