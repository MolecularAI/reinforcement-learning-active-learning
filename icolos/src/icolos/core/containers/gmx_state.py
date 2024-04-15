import numpy as np
import os
from typing import AnyStr, Any, Union, Optional
from pydantic import BaseModel
from icolos.core.containers.generic import GenericData
from icolos.utils.enums.program_parameters import GromacsEnum

from icolos.utils.enums.step_enums import StepGromacsEnum

_SGE = StepGromacsEnum()
_GE = GromacsEnum()


class AtomType(BaseModel):
    number: int
    a_type: str
    resi: int
    res: str
    atom: str
    cgnr: int
    charge: float
    mass: float


class GromacsState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    top_lines: list[str] = []
    itps: dict[str, str] = {}
    posre: dict[str, str] = {}
    atomtypes: list[AtomType] = []
    chains: Optional[list[str]] = None
    forcefield: str = "amber03"
    water: str = "tip3p"
    system: list[str] = []
    molecules: dict[str, Union[str, int]] = {}
    # hold the files as generic data objects, keyed by the index
    structures: dict[str, GenericData] = {}
    tprs: dict[str, GenericData] = {}
    trajectories: dict[str, GenericData] = {}
    log: dict[str, GenericData] = {}
    edr: dict[str, GenericData] = {}
    cpt: dict[str, GenericData] = {}
    ndx: list[GenericData] = []
    # store computed properties on the topology
    # {property: [val1, val2, val3]}
    properties: dict[str, list[Any]] = {}

    def __init__(self, **data) -> None:
        super().__init__(**data)

    def parse(self, path: str, file: str=_SGE.STD_TOPOL):
        """
        Populate the fields from parsing a topol file (normally from gmx pdb2gmx)
        If a moleculetype has been defined in a single topology, this is separated into its own itp file
        """
        # get the system name
        with open(os.path.join(path, file), "r") as f:
            lines = f.readlines()

        # first check for included atomtypes to be extracted to their own itp files
        start_idx = None
        stop_idx = None
        for line in lines:
            if line.startswith(_GE.MOLECULETYPES):
                start_idx = lines.index(line)
            # go all the way to IFDEF POSRE
            elif line.startswith("#ifdef POSRES") and start_idx is not None:
                stop_idx = lines.index(line) + 3
                break
        if all([x is not None for x in (start_idx, stop_idx)]):
            itp_extract = lines[start_idx:stop_idx]
            itp_key = itp_extract[2].split()[0] + ".itp"
            self.itps[itp_key] = itp_extract
            lines = lines[:start_idx] + lines[stop_idx:]

        # extract itp files (not including forcefield stuff)
        for line in lines:
            if (
                line.startswith("#include")
                and ".itp" in line
                and all([item not in line for item in (".ff", "posre")])
            ):
                itp_file = line.split()[-1].replace('"', "")
                with open(os.path.join(path, itp_file), "r") as f:
                    itp_lines = f.readlines()
                self.itps[itp_file] = itp_lines
        # extract water model
        for line in lines:
            if line.startswith("#include") and ".ff/tip" in line:
                self.water = line.split("/")[-1].split(".")[0].replace('"', "")
                self.forcefield = line.split()[-1].split(".")[0].replace('"', "")
        start = lines.index([l for l in lines if l.startswith(_GE.SYSTEM)][0])
        stop = lines.index([l for l in lines[start:] if l.startswith(_GE.MOLECULES)][0])
        if not self.system:
            for line in lines[start + 1 : stop]:
                if not line.startswith(";") and line.strip():
                    self.system.append(line.strip())
        # excract molecules and counts
        for line in lines[stop + 1 :]:
            if not line.startswith(";") and line.strip():
                parts = line.split()
                self.molecules[parts[0].strip()] = parts[1].strip()

        self.top_lines = self.generate_base_topol_file()

    def write_topol(self, base_dir: str, file: str="topol.top"):
        """
        Write a gromacs topology file, including its dependent itp files, to a dir
        """
        # update the base topol file
        self.generate_base_topol_file()
        with open(os.path.join(base_dir, file), "w") as f:
            for l in self.top_lines:
                f.write(l)
        for itp, lines in self.itps.items():
            if os.path.isfile(os.path.join(base_dir, itp)):
                os.remove(os.path.join(base_dir, itp))
            with open(os.path.join(base_dir, itp), "w") as f:
                for l in lines:
                    f.write(l)
        for itp, lines in self.posre.items():
            if os.path.isfile(os.path.join(base_dir, itp)):
                os.remove(os.path.join(base_dir, itp))
            with open(os.path.join(base_dir, itp), "w") as f:
                for l in lines:
                    f.write(l)

    def generate_base_topol_file(self):
        """
        Generates the main topology file
        """
        top_lines = []
        top_lines.append(f'#include "{self.forcefield}.ff/forcefield.itp"\n')
        # find any new atomtypes in the parametrised components, slot these in first
        if not self.atomtypes:
            self.atomtypes = self.collect_atomtypes()
        top_lines += self.atomtypes

        # now include the itp files
        for file in self.itps.keys():
            top_lines.append(f'#include "{file}"\n')
            # pdb2gmx appends posre control info to the bottom of any itp files
            if all([x not in file for x in ("Protein", "DNA", "RNA")]):
                # these are handled independently in the itp files
                top_lines.append(f'#ifdef POSRES\n#include "posre_{file}"\n#endif\n')

        # add water model
        top_lines.append(f'#include "{self.forcefield}.ff/{self.water}.itp"\n')
        top_lines.append(_SGE.WATER_POSRE)

        # add ions itp
        top_lines.append(f'#include "{self.forcefield}.ff/ions.itp"\n')

        top_lines.extend(self._construct_block(_GE.SYSTEM, self.system))
        top_lines.extend(self._construct_block(_GE.MOLECULES, self.molecules))
        self.top_lines = top_lines

    def add_itp(self, path, files: list[str], gen_posre: bool=True):
        for file in files:
            with open(os.path.join(path, file), "r") as f:
                lines = f.readlines()
            self.itps[file] = lines
            if gen_posre:
                # also generate a posre file
                self.generate_posre(path, file)

        self.generate_base_topol_file()

    def add_molecule(self, name: str, num: int=1):
        self.molecules[name] = num
        self.generate_base_topol_file()

    def add_posre(self, path, files: list[str]):
        for file in files:
            with open(os.path.join(path, file), "r") as f:
                lines = f.readlines()

            self.posre[file] = lines
        self.generate_base_topol_file()

    def generate_posre(self, path: str, itp_file: str, force: int=1000):
        stub = itp_file.split(".")[0]
        with open(os.path.join(path, itp_file), "r") as f:
            lines = f.readlines()
        start_idx = None
        stop_idx = None
        for line in lines:
            if line.startswith(_GE.ATOMS_DIRECTIVE):
                start_idx = lines.index(line) + 1
            elif line.startswith(_GE.BONDS) and start_idx is not None:
                stop_idx = lines.index(line)
                break
        lines = [
            l for l in lines[start_idx:stop_idx] if not l.startswith(";") and l.strip()
        ]
        args = ["number", "a_type", "resi", "res", "atom", "cgnr", "charge", "mass"]
        atoms = []
        for line in lines:
            args_dict = {}
            for arg, val in zip(args, line.split(";")[0].split()):
                args_dict[arg] = val
            atoms.append(AtomType(**args_dict))

        posre = "posre_" + stub + ".itp"
        out_path = os.path.join(path, posre)
        header = "\n[ position_restraints ]\n; atom  type      fx      fy      fz\n"
        written_lines = [header]
        with open(out_path, "w") as f:
            f.write(header)
            for atom in atoms:
                if not atom.a_type.upper().startswith("H"):
                    line = (
                        f"{atom.number:>6d}     1 {force:>5d} {force:>5d} {force:>5d}\n"
                    )
                    f.write(line)
                    written_lines.append(line)
        self.posre[posre] = written_lines

    def _construct_block(self, header: str, items: Union[dict[str, Any], Any]) -> list[str]:
        if not header.endswith("\n"):
            header += "\n"
        block = [header]
        if isinstance(items, dict):
            for key, value in items.items():
                block.append(f"{key}   {value}\n")
        else:
            for i in items:
                block.append(i + "\n")
        return block

    def collect_atomtypes(self) -> list[AtomType]:
        """
        Iterate over the itp files, strip any newly defined atomtypes, append to their own atomtypes.itp file, include this just after the forcefield include
        """
        atomtype_lines = []
        # read the topol file, identify all the itp files it is #including
        for itp, file_lines in self.itps.items():
            selection, remaining = self._extract_atomtype(file_lines)
            atomtype_lines.extend(selection)
            self.itps[itp] = remaining
        if atomtype_lines:
            atomtype_lines = self._remove_duplicate_atomtypes(atomtype_lines)
        return atomtype_lines

    def _extract_atomtype(self, lines: list[AnyStr]) -> list[AnyStr]:
        """
        Pull the atomtype lines out of the topol file and return them as a list, write the sanitised itp file to directory
        """

        start_index = None
        stop_index = None
        selection = []
        remaining = []
        for idx, line in enumerate(lines):
            if _GE.ATOMTYPES in line:
                start_index = idx
            if _GE.MOLECULETYPES in line:
                stop_index = idx
        if start_index is not None:
            selection = lines[start_index:stop_index]
            remaining = lines[:start_index]

        remaining.extend(lines[stop_index:])
        return selection, remaining

    def _remove_duplicate_atomtypes(self, atomtypes: list[AtomType]) -> list[AtomType]:
        output = [atomtypes[0]]
        for line in atomtypes:
            if line not in output:
                output.append(line)
        return output

    def set_topol(self, path: str, file: str=_SGE.STD_TOPOL):
        """
        When solvate or genion produce a modified topol, read this into the topology lines
        """
        with open(os.path.join(path, file), "r") as f:
            self.top_lines = f.readlines()

    def set_cpt(self, path: str, file: str=_SGE.STD_CPT, index: int=0):
        with open(os.path.join(path, file), "rb") as f:
            lines = f.read()

        cpt_file = GenericData(file_name=file, file_data=lines)
        self.cpt[index] = cpt_file

    def set_edr(self, path: str, file: str=_SGE.STD_EDR, index: int=0):
        with open(os.path.join(path, file), "rb") as f:
            lines = f.read()

        edr_file = GenericData(file_name=file, file_data=lines)
        self.edr[index] = edr_file

    def set_structure(self, path: str, file: str=_SGE.STD_STRUCTURE, sanitize: bool=False, index: int=0):
        with open(os.path.join(path, file), "r") as f:
            lines = f.readlines()
        if sanitize:
            lines = [l for l in lines if any([l.startswith(idx) for idx in _GE.ATOMS])]

        struct = GenericData(file_name=file, file_data=lines)
        self.structures[index] = struct

    def set_log(self, path: str, file: str=_SGE.STD_LOG, index: int=0):
        with open(os.path.join(path, file), "r") as f:
            lines = f.readlines()
        log = GenericData(file_name=file, file_data=lines)
        self.log[index] = log

    def set_tpr(self, path: str, file: str=_SGE.STD_TPR, index: int=0):
        with open(os.path.join(path, file), "rb") as f:
            data = f.read()
        data = GenericData(file_name=file, file_data=data, file_id=index)
        self.tprs[index] = data

    def write_tpr(self, path: str, file: str = _SGE.STD_TPR, index: int=0):
        tpr = self.tprs[index]
        tpr.write(os.path.join(path, file), join=False)

    def write_log(self, path: str, file: str = _SGE.STD_LOG, index: int=0):
        log = self.log[index]
        log.write(os.path.join(path, file), join=False)

    def write_cpt(self, path: str, file: str = _SGE.STD_CPT, index: int=0):
        cpt = self.cpt[index]
        cpt.write(os.path.join(path, file), join=False)
    
    def write_edr(self, path: str, file: str = _SGE.STD_EDR, index: int=0):
        edr = self.edr[index]
        edr.write(os.path.join(path, file), join=False)

    def set_ndx(self, path: str, file: str=_SGE.STD_INDEX):
        with open(os.path.join(path, file), "r") as f:
            self.ndx = f.readlines()

    def write_ndx(self, path: str, file: str=_SGE.STD_INDEX):
        with open(os.path.join(path, file), "w") as f:
            f.writelines(self.ndx)

    def write_structure(self, path: str, file: str=_SGE.STD_STRUCTURE, index: int=0):
        structure = self.structures[index]
        path = os.path.join(path, file)
        structure.write(path, join=False)

    def set_trajectory(self, path: str, file: str=_SGE.STD_XTC, index: int=0):
        # depending on mdp settings, some runs will not produce an xtc file, only trr
        if os.path.isfile(os.path.join(path, file)):
            with open(os.path.join(path, file), "rb") as f:
                data = f.read()
            data = GenericData(file_name=file, file_data=data, file_id=index)
            self.trajectories[index] = data

    def write_trajectory(self, path: str, file: str=_SGE.STD_XTC, index: int=0):
        traj = self.trajectories[index]

        path = os.path.join(path, file)
        traj.write(path, join=False)

    def append_structure(self, path: str, file: str, sanitize: bool=False, index: int=0):
        with open(os.path.join(path, file), "r") as f:
            lines = f.readlines()
        if sanitize:
            lines = [l for l in lines if any([l.startswith(idx) for idx in _GE.ATOMS])]
        data = self.structures[index].get_data() + lines
        self.structures[index].set_data(data)

    def write_props(self, path: str):
        """
        writes summary of properties for all trajectories to a file
        """
        prop_string = [f"{key}: {value}" for key, value in self.properties.items()]
        avg_prop_string = [
            f"{key}: {np.mean(value)}" for key, value in self.properties.items()
        ]

        output_string = f"""
ICOLOS PROPERTY SUMMARY
### Computed properties properties: {self.properties.keys()}

### Properties
{prop_string}


### Average Properties
{avg_prop_string} 
        """

        with open(os.path.join(path, "ICOLOS_PROPS.dat"), "w") as f:
            f.write(output_string)

    def __str__(self) -> str:
        return f"Gromacs Topology object: System: {self.system} | Molecules: {[m for m in self.molecules]} | FF: {self.forcefield} | itp files: {[f for f in self.itps.keys()]} | posre files {[f for f in self.posre.keys()]}"

    def __repr__(self) -> str:
        return self.__str__()
