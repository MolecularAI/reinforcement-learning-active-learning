from copy import deepcopy
from typing import Iterable, Literal, Optional, Any
from rdkit import Chem
from icolos.core.step_utils.obabel_structconvert import OBabelStructConvert

from icolos.utils.enums.compound_enums import (
    CompoundContainerEnum,
    EnumerationContainerEnum,
)
from icolos.utils.enums.program_parameters import SchrodingerExecutablesEnum
from icolos.core.step_utils.structconvert import StructConvert
from icolos.utils.general.icolos_exceptions import ContainerCorrupted
from icolos.utils.enums.write_out_enums import WriteOutEnum
from typing import Union
import numpy as np
import os

_WE = WriteOutEnum()


class Conformer:
    """This class is a storage class for individual conformers associated with a given Enumeration."""

    def __init__(
        self,
        conformer: Optional[Chem.Mol] = None,
        conformer_id: Optional[int] = None,
        enumeration_object: Optional["Enumeration"] = None,
    ):
        self._conformer = conformer
        self._conformer_id = conformer_id
        self._enumeration_object = enumeration_object
        self._extra_data_dictionary: dict[str, Any] = {}

    def has_molecule(self) -> bool:
        return isinstance(self._conformer, Chem.Mol)

    def get_compound_name(self) -> Optional[str]:
        if self.get_enumeration_object() is not None:
            return self.get_enumeration_object().get_compound_name()
        return None

    def get_index_string(self) -> str:
        enum_obj = self.get_enumeration_object()
        enum_str = ""
        if enum_obj is not None:
            enum_str = enum_obj.get_index_string()
        conf_str = ""
        if self.get_conformer_id() is not None:
            conf_str = str(self.get_conformer_id())
        return ":".join([enum_str, conf_str])

    def add_extra_data(self, key: str, data: Any):
        self._extra_data_dictionary[key] = data

    def get_extra_data(self) -> dict[str, Any]:
        return self._extra_data_dictionary

    def clear_extra_data(self):
        self._extra_data_dictionary = {}

    def set_enumeration_object(self, enumeration_object: "Enumeration"):
        self._enumeration_object = enumeration_object

    def get_enumeration_object(self):
        return self._enumeration_object

    def get_molecule(self) -> Optional[Chem.Mol]:
        return self._conformer
    
    def has_prop(self, tag: str) -> bool:
        mol = self.get_molecule()
        if mol is not None:
            return mol.HasProp(tag)
        return False
    
    def get_prop(self, tag: str) -> Optional[float]:
        mol = self.get_molecule()
        if mol is not None:
            return float(mol.GetProp(tag))
        return None
    
    def has_atomic_prop(self, tag: str) -> bool:
        mol = self.get_molecule()
        if mol is not None:
            atom = mol.GetAtomWithIdx(0)
            return atom.HasProp(tag)
        return False
    
    def get_atomic_prop(self, tag: str) -> Optional[np.ndarray]:
        mol = self.get_molecule()
        if mol is not None:
            return np.array([float(atom.GetProp(tag)) for atom in mol.GetAtoms()])
        return None
    
    def set_prop(self, tag: str, value: Any):
        if not self.has_molecule():
            raise ValueError("No molecule found for conformer")
        mol = self.get_molecule()
        if hasattr(value, "__len__"):
            # FIXME Dirty way of writing atom properties :(
            val = " ".join(str(v) for v in value)
            mol.SetProp(f"atom.dprop.{tag}", val)

            # This is how it *should* work, but rdkit doesn't want to write out atom properties
            # for atom, val in zip(mol.GetAtoms(), value):
            #     atom.SetProp(tag, str(val))
        else:
            mol.SetProp(tag, str(value))
    
    def get_any_prop(self, tag: str) -> Union[np.ndarray, float, None]:
        if self.has_prop(tag):
            return self.get_prop(tag)

        if self.has_atomic_prop(tag):
            return self.get_atomic_prop(tag)
        return None

    def set_molecule(self, conformer: Chem.Mol):
        self._conformer = conformer

    def set_conformer_id(self, conformer_id: int):
        self._conformer_id = conformer_id

    def get_conformer_id(self) -> Optional[int]:
        return self._conformer_id

    def empty(self) -> bool:
        return self.get_molecule() is None

    def _clone(self) -> "Conformer":
        clone = Conformer(
            conformer=deepcopy(self.get_molecule()),
            conformer_id=self.get_conformer_id(),
            enumeration_object=self.get_enumeration_object(),
        )
        clone._extra_data_dictionary = deepcopy(self.get_extra_data())
        return clone

    def __copy__(self) -> "Conformer":
        return self._clone()

    def __deepcopy__(self, memo: Any) -> "Conformer":
        return self._clone()

    def __repr__(self) -> str:
        parent_enumeration_id = (
            None
            if self.get_enumeration_object() is None
            else self.get_enumeration_object().get_enumeration_id()
        )
        return "<Icolos conformer: id=%s, parent enumeration: %s>" % (
            self.get_conformer_id(),
            parent_enumeration_id,
        )

    def __str__(self) -> str:
        return self.__repr__()

    def write(self, path: str, format_: str=_WE.SDF):
        writer = Chem.SDWriter(path)
        molecule = self.get_molecule()
        if molecule is None:
            raise ValueError(f"No rdkit molecule object to write")
        molecule.SetProp(_WE.RDKIT_NAME, self.get_index_string())
        molecule.SetProp(_WE.INDEX_STRING, self.get_index_string())
        writer.write(molecule)
        writer.close()
        if format_ == _WE.PDB:
            pdb_path = path.split(".")[0] + ".pdb"
            # convert the written sdf file to a pdb with OB
            converter = OBabelStructConvert()
            converter.sdf2pdb(sdf_file=path, pdb_file=pdb_path)
            os.remove(path)

    def update_coordinates(self, path: str):
        old = self.get_molecule()
        if old is None:
            raise ValueError(f"No rdkit molecule object to update")
        for mol in Chem.SDMolSupplier(path, removeHs=False):
            mol.SetProp(_WE.RDKIT_NAME, old.GetProp(_WE.RDKIT_NAME))
            for prop in old.GetPropNames():
                mol.SetProp(prop, old.GetProp(prop))
            self.set_molecule(mol)

            # only one molecule expected at this stage, so stop after first run
            break
        self.write("".join([path, "_out"]))


class Enumeration:
    """This class bundles all information on an enumeration, especially all conformers generated."""

    def __init__(
        self,
        compound_object: Optional["Compound"] = None,
        smile: str = "",
        molecule: Optional[Chem.Mol] = None,
        original_smile: Optional[str] = None,
        enumeration_id: Optional[int] = None,
    ):
        self._MC = CompoundContainerEnum()
        self._EC = EnumerationContainerEnum()
        self._smile = smile
        self._compound_object = compound_object
        self._molecule = molecule
        self._original_smile = original_smile
        self._enumeration_id = enumeration_id
        self._conformers: list[Conformer] = []

    def empty(self) -> bool:
        return len(self.get_conformers()) == 0
    
    def has_molecule(self) -> bool:
        return isinstance(self._molecule, Chem.Mol)

    def get_compound_name(self) -> Optional[str]:
        if self.get_compound_object() is not None:
            return self.get_compound_object().get_name()
        return None

    def _get_next_conformer_id(self) -> int:
        ids = [conf.get_conformer_id() for conf in self.get_conformers()]
        valid_ids: list[int] = [id for id in ids if id is not None]
        if len(valid_ids) > 0:
            return max(valid_ids) + 1
        return 0

    def sort_conformers(
        self, by_tag: Union[str, list[str]], reverse: bool=True, aggregation: Literal["sum", "product"]="sum"
    ):
        conformers = self.get_conformers()
        if isinstance(by_tag, str):
            if conformers[0].get_prop(by_tag) is None:
                raise ValueError(f"Could not find tag {by_tag}")
            conformers = sorted(
                conformers,
                key=lambda x: x.get_prop(by_tag),
                reverse=reverse,
            )
            self._conformers = conformers
            self.reset_conformer_ids()
        elif isinstance(by_tag, list):
            # need to normalise the values, calculate max and min of each tag in the series
            def normalise_tag(value: float, tag: str) -> float:
                all_tag_values = [
                    float(conf.get_molecule().GetProp(tag)) for conf in conformers
                ]
                max_tag = np.max(all_tag_values)
                min_tag = np.min(all_tag_values)
                return (float(value) - min_tag) / (max_tag - min_tag)

            # if we specify multiple tags, aggregate according the the provided aggregation function
            if aggregation == "sum":
                conformers = sorted(
                    conformers,
                    key=lambda x: np.sum(
                        [
                            float(normalise_tag(x.get_molecule().GetProp(i), i))
                            for i in by_tag
                        ]
                    ),
                    reverse=reverse,
                )
                self._conformers = conformers
            elif aggregation == "product":
                conformers = sorted(
                    conformers,
                    key=lambda x: np.product(
                        [
                            float(normalise_tag(x.get_molecule().GetProp(i), i))
                            for i in by_tag
                        ]
                    ),
                    reverse=reverse,
                )
                self._conformers = conformers
            else:
                raise AttributeError(
                    "Only sum or product aggregation modes are currently supported - ABORT"
                )
                # for ligand in self.ligands:

    #    ligand.set_conformers(sorted(ligand.get_conformers(),
    #                                 key=lambda x: float(x.GetProp(_ROE.GLIDE_DOCKING_SCORE)), reverse=False))
    #    ligand.add_tags_to_conformers()

    def find_conformer(self, conformer_id: int) -> Conformer:
        conf = [
            conf
            for conf in self.get_conformers()
            if conf.get_conformer_id() == conformer_id
        ]
        if len(conf) == 0:
            raise IndexError(f"Could not find conformer with id {conformer_id}.")
        elif len(conf) > 1:
            raise ContainerCorrupted(
                f"More than one conformer with id {conformer_id} found in the same Enumeration instance (compound_number: {self.get_enumeration_id()})."
            )
        return conf[0]

    def get_conformer_ids(self) -> list[int]:
        ids = [conf.get_conformer_id() for conf in self.get_conformers()]
        return [id for id in ids if id is not None]

    def reset_conformer_ids(self):
        for new_id, conf in enumerate(self.get_conformers()):
            conf.set_conformer_id(conformer_id=new_id)

    def add_conformer(self, conformer: Conformer, auto_update: bool=True):
        """Add a new conformer. If "auto_update" is True, the Enumeration class will be set to "self" and
        the conformer_id will be set to the next free index."""
        conformer = deepcopy(conformer)
        if auto_update:
            conformer.set_enumeration_object(self)
            conformer.set_conformer_id(self._get_next_conformer_id())
        self._conformers.append(conformer)

    def add_conformers(self, conformers: list[Conformer], auto_update: bool=True):
        """Add new conformers. If "auto_update" is True, the Enumeration class will be set to "self" and
        the conformer_id will be set to the next free index."""
        for conformer in conformers:
            self.add_conformer(conformer=conformer, auto_update=auto_update)

    def get_index_string(self) -> str:
        comp_obj = self.get_compound_object()
        comp_str = ""
        if comp_obj is not None:
            comp_str = comp_obj.get_index_string()
        enum_str = ""
        if self.get_enumeration_id() is not None:
            enum_str = str(self.get_enumeration_id())
        return ":".join([comp_str, enum_str])

    def clean_failed_conformers(self):
        # all conformers, where the molecule has been set to None by a function can be considered to have failed
        for idx in list(reversed(range(len(self._conformers)))):
            if self._conformers[idx].get_molecule() is None:
                del self._conformers[idx]
        self.reset_conformer_ids()

    def clear_molecule(self):
        self._molecule = None

    def clear_conformers(self):
        self._conformers = []

    def get_conformers(self) -> list[Conformer]:
        return self._conformers

    def clone_conformers(self) -> list[Conformer]:
        return [deepcopy(conf) for conf in self._conformers]

    def set_compound_object(self, compound_object: "Compound"):
        self._compound_object = compound_object

    def get_compound_object(self):
        return self._compound_object

    def set_enumeration_id(self, enumeration_id: int):
        self._enumeration_id = enumeration_id

    def get_enumeration_id(self) -> Optional[int]:
        return self._enumeration_id

    def set_smile(self, smile: str):
        self._smile = smile

    def get_smile(self) -> str:
        return self._smile

    def set_molecule(self, molecule: Chem.Mol):
        self._molecule = molecule

    def get_molecule(self) -> Chem.Mol:
        return self._molecule

    def set_original_smile(self, original_smile: str):
        self._original_smile = original_smile

    def get_original_smile(self) -> Optional[str]:
        return self._original_smile

    def _clone(self) -> "Enumeration":
        clone = Enumeration(
            compound_object=self.get_compound_object(),
            smile=self.get_smile(),
            molecule=deepcopy(self.get_molecule()),
            original_smile=self.get_original_smile(),
            enumeration_id=self.get_enumeration_id(),
        )
        for conf in self.get_conformers():
            conf = deepcopy(conf)
            conf.set_enumeration_object(enumeration_object=clone)
            clone.add_conformer(conf, auto_update=False)
        return clone

    def __copy__(self) -> "Enumeration":
        return self._clone()

    def __deepcopy__(self, memo: Any) -> "Enumeration":
        return self._clone()

    def __repr__(self) -> str:
        parent_compound_id = (
            None
            if self.get_compound_object() is None
            else self.get_compound_object().get_compound_number()
        )
        return (
            "<Icolos enumeration: id=%s, smile=%s, parent compound: %s, num_conformers: %i>"
            % (
                self.get_enumeration_id(),
                self.get_smile(),
                parent_compound_id,
                len(self._conformers),
            )
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self) -> Iterable[Conformer]:
        return iter(self._conformers)

    def __getitem__(self, key: int) -> Conformer:
        return self._conformers[key]

    def __len__(self) -> int:
        return len(self.get_conformers())


class Compound:
    """This class bundles all information on a molecule and serves mainly to group enumerations."""

    def __init__(self, name: str = "", compound_number: Optional[int] = None):
        self._CC = CompoundContainerEnum()
        self._EC = EnumerationContainerEnum()
        self._name = name
        self._compound_number = compound_number
        self._enumerations: list[Enumeration] = []

    def __repr__(self) -> str:
        return "<Icolos compound: name=%s, compound_number=%s, enumerations=%s>" % (
            self.get_name(),
            self.get_compound_number(),
            len(self.get_enumerations()),
        )

    def __str__(self) -> str:
        return self.__repr__()

    def get_index_string(self) -> str:
        if self.get_compound_number() is not None:
            return str(self.get_compound_number())
        return ""

    def set_name(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name

    def set_compound_number(self, compound_number: int):
        self._compound_number = compound_number

    def get_compound_number(self) -> int:
        return self._compound_number

    def add_enumeration(self, enumeration: Enumeration, auto_update: bool=True):
        """Add a new enumeration. If "auto_update" is True, the Compound class will be set to "self" and
        the enumeration_id will be set to the next free index."""
        enumeration = deepcopy(enumeration)
        if auto_update:
            enumeration.set_compound_object(self)
            enumeration.set_enumeration_id(self._get_next_enumeration_id())
        self._enumerations.append(enumeration)

    def add_enumerations(self, enumerations: list[Enumeration], auto_update: bool=True):
        """Add new enumerations. If "auto_update" is True, the Compound class will be set to "self" and
        the enumeration_id will be set to the next free index."""
        for enumeration in enumerations:
            self.add_enumeration(enumeration=enumeration, auto_update=auto_update)

    def clear_enumerations(self):
        self._enumerations = []

    def find_enumeration(self, idx: int) -> Enumeration:
        for enum in self.get_enumerations():
            if enum.get_enumeration_id() == idx:
                return enum

    def get_enumerations(self) -> list[Enumeration]:
        return self._enumerations

    def _clone(self) -> "Compound":
        clone = Compound(
            name=self.get_name(), compound_number=self.get_compound_number()
        )
        for enum in self.get_enumerations():
            enum = deepcopy(enum)
            enum.set_compound_object(compound_object=clone)
            clone.add_enumeration(enum, auto_update=False)
        return clone

    def __iter__(self) -> Iterable[Enumeration]:
        return iter(self._enumerations)

    def __copy__(self) -> "Compound":
        return self._clone()

    def __deepcopy__(self, memo: Any) -> "Compound":
        return self._clone()

    def __getitem__(self, key: int) -> Enumeration:
        return self._enumerations[key]

    def __len__(self) -> int:
        return len(self.get_enumerations())

    def _get_next_enumeration_id(self):
        ids = [enum.get_enumeration_id() for enum in self.get_enumerations()]
        if len(ids) == 0:
            return 0
        return max(ids) + 1

    def find_enumeration(self, enumeration_id: int) -> Enumeration:
        enum = [
            enum
            for enum in self.get_enumerations()
            if enum.get_enumeration_id() == enumeration_id
        ]
        if len(enum) == 0:
            raise IndexError(f"Could not find enumeration with id {enumeration_id}.")
        elif len(enum) > 1:
            raise ContainerCorrupted(
                f"More than one enumeration with id {enumeration_id} found in the same Compound instance (compound_number: {self.get_compound_number()})."
            )
        return enum[0]

    def get_enumeration_ids(self) -> list[int]:
        ids = [enum.get_enumeration_id() for enum in self.get_enumerations()]
        return ids

    def reset_enumeration_ids(self):
        for new_id, enum in enumerate(self.get_enumerations()):
            enum.set_enumeration_id(enumeration_id=new_id)

    def reset_all_ids(self):
        self.reset_enumeration_ids()
        for enum in self.get_enumerations():
            enum.reset_conformer_ids()

    def update_all_relations(self):
        for enum in self.get_enumerations():
            enum.set_compound_object(self)
            for conf in enum.get_conformers():
                conf.set_enumeration_object(enum)

    def empty(self) -> bool:
        return len(self.get_enumerations()) == 0

    def unroll_conformers(self) -> list[Conformer]:
        conformers = []
        for enum in self.get_enumerations():
            # guard against empty enumerations that might be used when constructing more complex data flows
            if enum.empty():
                continue
            for conf in enum.get_conformers():
                conformers.append(conf)
        return conformers


# TODO: Replacing these three functions by a wrapper object
def get_compound_by_id(compounds: list[Compound], id: int) -> Compound:
    for compound in compounds:
        if compound.get_compound_number() == id:
            return compound
    raise ValueError(
        f"Could not find compound with id {id} in list of length {len(compounds)}."
    )


def get_compound_by_name(compounds: list[Compound], name: str) -> Compound:
    for compound in compounds:
        if compound.get_name() == name:
            return compound
    raise ValueError(
        f"Could not find compound with name {name} in list of length {len(compounds)}."
    )


def unroll_conformers(compounds: list[Compound]) -> list[Conformer]:
    all_conformers = []
    for comp in compounds:
        all_conformers = all_conformers + comp.unroll_conformers()
    return all_conformers


def unroll_enumerations(compounds: list[Compound]) -> list[Enumeration]:
    all_enumerations = []
    for comp in compounds:
        all_enumerations = all_enumerations + comp.get_enumerations()
    return all_enumerations
