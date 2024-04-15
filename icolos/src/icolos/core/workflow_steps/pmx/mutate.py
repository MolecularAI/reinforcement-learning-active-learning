""" Mutate - Protein FEP by residue mutation """

import os

from pydantic import BaseModel

from icolos.core.containers.perturbation_map import Edge, Node, PerturbationMap
from icolos.core.workflow_steps.pmx.base import StepPMXBase
from icolos.utils.enums.logging_enums import LoggingConfigEnum
from icolos.utils.enums.program_parameters import GromacsEnum, PMXAtomMappingEnum, PMXEnum
from icolos.utils.enums.step_enums import StepGromacsEnum
from icolos.utils.execute_external.execute import Executor

_SGE = StepGromacsEnum()
_GE = GromacsEnum()
_PE = PMXEnum()
_PAE = PMXAtomMappingEnum()
_LE = LoggingConfigEnum()


class StepPMXmutate(StepPMXBase, BaseModel):
    """
    This is the de facto entrypoint for a protein FEP workflow.
    This step does the legwork for setting up the output dir structure,
    similar to pmx_setup for the small molecule case.

    Input
    -----
    :generic pdb: Protein apo input structure
    :generic mut: List of mutations
    :generic mdp:
        Directory containing Gromacs mdp configuration files,
        using the following names: "em", "nvt", "eq", "ti"
    :generic top: Protein topology, optional

    Requires
    --------
    :work_dir dir: Working directory containing output of previous steps, optional

    Settings
    --------
    :additional forcefield: force field to use, default = amber99sb-star-ildn-mut.ff
    :additional water: water model, default = tip3p
    :additional replicas: number of individual runs to perform for each edge, default = 3

    """

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_backend(Executor)

    def execute(self):
        # start with a pdb file, parametrize with pdb2gmx
        replicas = self.settings.additional.get("replicas", 3)
        assert self.work_dir is not None

        # this is a bit awkward putting them in the ligand dir, we adopt the same structure as the small molecule approach
        os.makedirs(os.path.join(self.work_dir, "input"), exist_ok=True)
        for folder in ["ligands", "mdp", "protein"]:
            os.makedirs(os.path.join(self.work_dir, "input", folder), exist_ok=True)
        setup_dir = os.path.join(self.work_dir, _PAE.LIGAND_DIR)
        self.data.generic.write_out_all_files(setup_dir)

        mdp_dir = self.data.generic.get_argument_by_extension(
            ext="mdp", rtn_file_object=True
        )
        mdp_dir.write(os.path.join(self.work_dir, "input/mdp"))

        # run pdb2gmx on the protein, generate a good input file
        pdb2gmx_args = [
            "-f",
            self.data.generic.get_argument_by_extension("pdb"),
            "-water",
            self.settings.additional["water"],
            "-ff",
            self.settings.additional["forcefield"],
            "-o",
            # use pdb here to retain chain IDs for pmx mutate
            os.path.join(setup_dir, "wt.pdb"),
            "-ignh",
        ]
        # run the
        self._gromacs_executor.execute(
            command=_GE.PDB2GMX, arguments=pdb2gmx_args, check=True, location=setup_dir
        )
        # now we have the prepared structure to pass to pmx mutate

        # parse mutant file, note that pdb2gmx will remove chain IDs
        with open(
            os.path.join(setup_dir, self.data.generic.get_argument_by_extension("mut")),
            "r",
        ) as f:
            mut_lines = [l for l in f.readlines() if l.strip()]

        perturbation_map = PerturbationMap()
        wt_node = Node(node_id="wt", node_hash="wt")
        perturbation_map.add_node(wt_node)
        # all we need is the identities of the mutants

        # in the form "P 76 TYR\n X 45 GLY"
        for mut in mut_lines:
            chain, resid, new_res = mut.split()
            edge_dir = os.path.join(self.work_dir, f"wt_{chain}{resid}{new_res}")
            os.makedirs(
                edge_dir,
                exist_ok=True,
            )
            for branch in self.therm_cycle_branches:
                for state in self.states:
                    for r in range(1, replicas + 1):
                        for sim in self.sim_types:
                            os.makedirs(
                                os.path.join(
                                    self.work_dir,
                                    edge_dir,
                                    branch,
                                    state,
                                    f"run{r}",
                                    sim,
                                ),
                                exist_ok=True,
                            )
            mutate_args = [
                "-f",
                os.path.join(setup_dir, "wt.pdb"),
                "--ref",
                os.path.join(setup_dir, "wt.pdb"),
                "-o",
                f"{chain}{resid}{new_res}.pdb",
                "-ff",
                self._get_additional_setting(_SGE.FORCEFIELD),
            ]
            self._logger.log(
                f"Generating mutation {chain}{resid}->{new_res}", _LE.DEBUG
            )
            result = self._backend_executor.execute(
                command=_PE.MUTATE,
                arguments=mutate_args,
                location=os.path.join(edge_dir, "bound"),
                check=True,
                pipe_input=f'echo -e "{chain}\n{resid}\n{new_res}\nn"',
            )

            # run pdb2gmx on the resulting structure, resulting structures are for the bound states

            pdb2gmx_args = [
                "-f",
                f"{chain}{resid}{new_res}.pdb",
                "-water",
                self.settings.additional["water"],
                "-ff",
                self.settings.additional["forcefield"],
                "-o",
                "init.pdb",
            ]
            self._backend_executor.execute(
                command=_GE.PDB2GMX,
                arguments=pdb2gmx_args,
                location=os.path.join(edge_dir, "bound"),
                check=True,
            )
            # now make the directory in the main workdir to  for that mutation and copy the input files over

            # now we have a single mutated structure file per system.
            node = Node(
                node_id=f"{chain}{resid}{new_res}",
                node_hash=f"{chain}{resid}{new_res}",
            )
            perturbation_map.add_node(node)
            perturbation_map.add_edge(Edge(node_from=wt_node, node_to=node))

            # generate connectivity (just a star map), now we have an initialized perturbation map, relying on file paths only, not attached structures for now
            for node in perturbation_map.nodes:
                perturbation_map._attach_node_connectivity(node)

            self.get_workflow_object().set_perturbation_map(perturbation_map)

            # TODO: we need to generate mutatinos for the bound state and the unbound state, by separating the pdb file

            # we have the chain being mutated, extract this to its own file

            # create the index file
            make_ndx_args = ["-f", "init.pdb"]
            self._backend_executor.execute(
                command=_GE.MAKE_NDX,
                arguments=make_ndx_args,
                location=os.path.join(edge_dir, "bound"),
                pipe_input=f'echo -e "chain {chain}\nq"',
                check=True,
            )
            # index.ndx in the bound state directory

            editconf_args = [
                "-f",
                "init.pdb",
                "-n",
                "index.ndx",
                "-o",
                os.path.join(
                    edge_dir, "unbound", f"{chain}{resid}{new_res}_unbound.pdb"
                ),
            ]
            self._backend_executor.execute(
                command=_GE.EDITCONF,
                arguments=editconf_args,
                location=os.path.join(edge_dir, "bound"),
                check=True,
                pipe_input=f'echo -e "ch{chain}\n"',
            )

            # generate topology for the unbound state
            pdb2gmx_args = [
                "-f",
                f"{chain}{resid}{new_res}_unbound.pdb",
                "-water",
                self.settings.additional["water"],
                "-ff",
                self.settings.additional["forcefield"],
                "-o",
                "init.pdb",
            ]
            self._backend_executor.execute(
                command=_GE.PDB2GMX,
                arguments=pdb2gmx_args,
                location=os.path.join(edge_dir, "unbound"),
                check=True,
            )


help_string = """
pmx mutate -h
usage: pmx [-h] [-f infile] [-fB infileB] [-o outfile] [-ff ff]                                    
           [--script script] [--keep_resid | --ref ] [--resinfo]                                   
                                                                                                   
This script applies mutations of residues in a structure file for subsequent                                          
free energy calculations. It supports mutations to protein, DNA, and RNA                           
molecules.                                                                                         
                                                                          
The mutation information and dummy placements are taken from the hybrid residue
database "mutres.mtp". The best way to use this script is to take a pdb/gro file                                                                    
that has been written with pdb2gmx with all hydrogen atoms present.
                                                                          
By default, all residues are renumbered starting from 1, so to have unique
residue IDs. If you want to keep the original residue IDs, you can use the flag                                                                     
--keep_resid. In this case, you will also need to provide chain information                                                                         
in order to be able to mutate the desired residue. Alternatively, if you would                                                                      
like to use the original residue IDs but these have been changed, e.g. by gromacs,
you can provide a reference PDB file (with chain information too) using the --ref
flag. The input structure will be mutated according to the IDs chosen for the
reference structure after having mapped the two residue indices.

The program can either be executed interactively or via script. The script file
simply has to consist of "residue_id target_residue_name" pairs (just with some
space between the id and the name), or "chain_id residue_id target_residue_name"
if you are keeping the original residue IDs or providing a reference structure.

The script uses an extended one-letter code for amino acids to account for
different protonation states. Use the --resinfo flag to print the dictionary.

optional arguments:
  -h, --help       show this help message and exit
  -f infile        Input structure file in PDB or GRO format. Default is "protein.pdb"
  -fB infileB      Input structure file of the B state in PDB or GRO format (optional).
  -o outfile       Output structure file in PDB or GRO format. Default is "mutant.pdb"
  -ff ff           Force field to use. If none is provided, 
                   a list of available ff will be shown.
  --script script  Text file with list of mutations (optional).
  --keep_resid     Whether to renumber all residues or to keep the
                   original residue IDs. By default, all residues are
                   renumbered so to have unique IDs. With this flags set,
                   the original IDs are kept. Because the IDs might not
                   be unique anymore, you will also be asked to choose
                   the chain ID where the residue you want to mutate is.
  --ref            Provide a reference PDB structure from which to map
                   the chain and residue IDs onto the file to be mutated (-f).
                   This can be useful when wanting to mutate a file that
                   has had its residues renumbered or the chain information
                   removed (e.g. after gmx grompp). As in the --keep_resid
                   option, if --ref is chosen, you will need to provide chain
                   information either interactively or via the --script flag.
  --resinfo        Show the list of 3-letter -> 1-letter residues
"""
