from typing import Dict, List, Optional
import uuid
from IPython.lib.display import IFrame
import pandas as pd
from icolos.core.containers.compound import Compound, Conformer
from pyvis.network import Network
from icolos.core.containers.generic import GenericData
from icolos.loggers.iologger import IOLogger
from icolos.utils.enums.parallelization import ParallelizationEnum
from icolos.utils.enums.logging_enums import LoggingConfigEnum
from icolos.utils.enums.step_enums import StepFepPlusEnum
import os
from pydantic import BaseModel, PrivateAttr


_LE = LoggingConfigEnum()
_SFE = StepFepPlusEnum()
_PE = ParallelizationEnum


class Node(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    node_id: Optional[str] = None
    node_hash: Optional[str] = None
    conformer: Optional[Conformer] = None
    node_connectivity: list["Edge"] = []

    def __init__(self, **data) -> None:
        super().__init__(**data)

    def get_node_id(self) -> str:
        return self.node_id

    def get_node_color(self) -> str:
        # TODO: Expand this so we have different colours for each connectivity number [1,10]
        # this is just a placeholder for now
        thresholds = {i: "c0affe" for i in range(10)}

        num_connections = len(self.node_connectivity)
        return thresholds[num_connections]

    def set_node_id(self, node_id: str):
        self.node_id = node_id

    def get_conformer(self) -> Conformer:
        return self.conformer

    def set_conformer(self, conformer: Conformer):
        self.conformer = conformer

    def get_node_hash(self) -> str:
        return self.node_hash


class Edge(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    node_from: Node = Node()
    node_to: Node = Node()
    total: Optional[str] = None
    mcs: Optional[str] = None
    chg: Optional[str] = None
    softbond: Optional[str] = None
    min_no_atoms: Optional[str] = None
    snapCoreRmsd: Optional[str] = None
    bidirSnapCoreRmsd: Optional[str] = None
    status: _PE = _PE.STATUS_SUCCESS
    ddG: float = 0.0
    ddG_err: float = 0.0

    def __init__(self, **data):
        super().__init__(**data)

    def _get_source_node_name(self) -> str:
        return self.node_from.get_node_hash()

    def _get_destination_node_name(self) -> str:
        return self.node_to.get_node_hash()

    def get_edge_id(self) -> str:
        # construct the edge ID from the node hashes, separated by '_'
        return f"{self.node_from.get_node_hash()}_{self.node_to.get_node_hash()}"

    def set_status(self, status: str):
        assert status in [_PE.STATUS_SUCCESS, _PE.STATUS_FAILED]
        self.status = status


class PerturbationMap(BaseModel):
    """Hold a map construction parsed from a csv (probabably from a parsed schrodinger log
    file or something) and provide some utility methods for doing pmx calculations on the edges"""

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    nodes: list[Node] = []
    edges: list[Edge] = []
    hash_map: dict[str, Node] = {}
    compounds: list[Compound] = []
    protein: GenericData = None
    vmap_output: IFrame = None
    replicas: int = 3
    node_df: pd.DataFrame = None
    # prune subsequent edge calculations on error
    strict_execution: str = False
    hub_conformer: Conformer = None
    _logger = PrivateAttr()

    def __init__(self, **data) -> None:
        self._logger = IOLogger()
        super().__init__(**data)

    def _get_line_idx(self, data: list[str], id_str: str) -> int:
        line = [e for e in data if id_str in e]
        assert len(line) == 1
        line = line[0]
        return data.index(line)

    def _get_conformer_by_id(self, comp_name: str=None, comp_idx: int=None) -> Conformer:
        """Parses the compound names from the edges specified in the mapper log file, performs a lookup to extract the corresponding compound from the step's compounds

        :param str comp_id: id of the compound parsed from the map generation
        :return Conformer: return the conformer object from self.data.compounds
        """
        self._logger.log(f"Searching for conformer ID {comp_name} in {[c.get_name() for c in self.compounds]}", _LE.DEBUG)
        for compound in self.compounds:
            # compound could have any arbitrary name - check whether this the node ID from schrodinger, or check the index string
            if (
                compound.get_name() == comp_name
                or compound.get_compound_number() == comp_idx
            ):
                # there will be one and only one conformer attached to one of the enumerations.  Return the first.
                for enum in compound.get_enumerations():
                    if enum.get_conformers():
                        return enum.get_conformers()[0]
        raise ValueError(f"No conformer with ID {comp_name} or index {comp_idx} found")

    def generate_from_lomap_output(self, file: str):
        output_table = pd.read_csv(file, delimiter="   ,", index_col=False)
        output_table = output_table[output_table["Connect"].str.contains("Yes") == True]
        output_table.set_axis(
            [
                "Index_1",
                "Index_2",
                "Filename_1",
                "Filename_2",
                "Str_sim",
                "Eff_sim",
                "Loose_sim",
                "Connect",
            ],
            axis=1,
            inplace=True,
        )
        # generate a unique hash for each compound number, then we have mapping from compound number to node hash
        compound_df = pd.read_csv("out.txt", delimiter="\t", index_col="#ID")
        unique_hashes = [uuid.uuid4().hex for _ in compound_df.FileName]
        compound_df["NodeHash"] = unique_hashes

        # construct hash_ids for each edge
        hash_ids = []
        for node_from, node_to in zip(
            output_table["Filename_1"], output_table["Filename_2"]
        ):
            node_from = node_from.strip()
            node_to = node_to.strip()
            hash_from = compound_df[
                compound_df["FileName"] == node_from
            ].NodeHash.values[0]
            hash_to = compound_df[compound_df["FileName"] == node_to].NodeHash.values[0]
            hash_ids.append(f"{hash_from}_{hash_to}")
        output_table["EdgeHash"] = hash_ids

        for _, row in compound_df.iterrows():
            compound_id = row.FileName.split(".")[0].split(":")[0]
            self.nodes.append(
                Node(
                    node_id=compound_id,
                    node_hash=row.NodeHash,
                    conformer=self._get_conformer_by_id(comp_idx=int(compound_id)),
                )
            )

        for _, edge in output_table.iterrows():
            edge = Edge(
                node_from=self._get_node_by_hash_id(edge.EdgeHash.split("_")[0]),
                node_to=self._get_node_by_hash_id(edge.EdgeHash.split("_")[1]),
                mcs=float(edge.Str_sim),
            )
            self.edges.append(edge)

        for node in self.nodes:
            self._attach_node_connectivity(node)

    def generate_star_map(self):
        """Generates a star topology using a single hub compound"""
        hub_node = Node(
            node_id=self.hub_compound.get_index_string(),
            node_hash=uuid.uuid4().hex,
            conformer=self.hub_compound.get_molecule(),
        )
        for compound in self.compounds:
            end_node = Node(
                node_id=compound.get_index_string(),
                node_hash=uuid.uuid4().hex,
                conformer=compound.get_enumerations()[0]
                .get_conformers()[0]
                .get_molecule(),
            )
            edge = Edge(
                node_from=hub_node,
                node_to=end_node,
            )
            self.edges.append(edge)

        for node in self.nodes:
            self._attach_node_connectivity(node)

    def parse_map_file(self, file_path: str):
        """Parse map from Schrodinger's fep_mapper log file, build internal graph representation + attach properties from fmp_stats, if provided

        :param str file_path: path to the fep_mapper.log file to extract the perturbation map from
        """
        # we need to do some format enforcement here (schrodinger or otherwise)

        with open(file_path, "r") as f:
            data = f.readlines()

        start_node = self._get_line_idx(data, _SFE.NODE_HEADER_LINE)
        stop_node = self._get_line_idx(data, _SFE.SIMULATION_PROTOCOL)
        edge_info_start = self._get_line_idx(data, _SFE.SIMILARITY)

        # TODO: refactor that
        # clean up the data from schrodinger
        split_data = []
        for line in data:
            split_line = line.split("  ")
            stripped_line = []
            for element in split_line:
                if not element.isspace() and element:
                    stripped_line.append(element.strip())
            split_data.append(stripped_line)

        data = split_data

        self.node_df = pd.DataFrame(
            data[start_node + 3 : stop_node - 1],
            index=None,
            columns=[
                "hash_id",
                "node_id",
                "Predicted dG",
                "Experimental dG",
                "Predicted Solvation dG",
                "Experimental Solvation dG",
            ],
        )
        edge_info = pd.DataFrame(
            data[edge_info_start + 3 : -1],
            columns=[
                "Short ID",
                "Total",
                "Mcs",
                "Charge",
                "SoftBond",
                "MinimumNumberOfAtom",
                "SnapCoreRmsd",
                "BidirectionSnapCore",
            ],
        ).dropna()
        for hash_id, node_id in zip(self.node_df["hash_id"], self.node_df["node_id"]):
            # map the hashes to the compound IDs
            self.hash_map[hash_id] = node_id
            node = Node(
                node_id=node_id,
                node_hash=hash_id,
                conformer=self._get_conformer_by_id(node_id),
            )
            self._logger.log(f"Created node {node}", _LE.DEBUG)
            # generate the Node object to wrap the compound
            self.nodes.append(node)

        for _, edge in edge_info.iterrows():
            edge = Edge(
                node_from=self._get_node_by_hash_id(edge[0].split("_")[0]),
                node_to=self._get_node_by_hash_id(edge[0].split("_")[1]),
                total=edge[1],
                mcs=edge[2],
                chg=edge[3],
                softbond=edge[4],
                min_no_atoms=edge[5],
                snapCoreRmsd=edge[6],
                bidirSnapCoreRmsd=edge[7],
            )
            self.edges.append(edge)
        # process the node info, generate the hash map
        for node in self.nodes:
            self._attach_node_connectivity(node)

    def _attach_node_connectivity(self, node: Node):
        # looks through the constructed edges, returns ids of any edges that have the specified node as one component
        connected_edges = []
        for edge in self.edges:
            if (
                edge.node_from.get_node_hash() == node.node_hash
                or edge.node_to.get_node_hash() == node.node_hash
            ):
                connected_edges.append(edge.get_edge_id())
        node.node_connectivity = connected_edges

    def _get_node_by_node_id(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.node_id == node_id:
                return node

    def _get_node_by_hash_id(self, hash_id: str) -> Node:
        for node in self.nodes:
            if node.node_hash == hash_id:
                return node

    def get_edges(self, alive_only=True) -> list[Edge]:
        if alive_only:
            return [e for e in self.edges if e.status == _PE.STATUS_SUCCESS]
        return self.edges

    def get_nodes(self) -> list[Node]:
        return self.nodes

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def add_node(self, node: Node):
        self.nodes.append(node)

    def visualise_perturbation_map(self, write_out_path: str):
        """Generate NetworkX graph for the parsed perturbation map

        :param str write_out_path: directory to write output file
        """
        vmap = Network(directed=True)
        vmap.barnes_hut()

        # this is not an iterable
        for edge in self.edges:
            vmap.add_node(
                edge._get_source_node_name(), color=edge.node_from.get_node_color()
            )
            vmap.add_node(
                edge._get_destination_node_name(), color=edge.node_to.get_node_color()
            )
            vmap.add_edge(
                source=edge._get_source_node_name(),
                to=edge._get_destination_node_name(),
                length=edge.total,
                label="total: " + str(edge.total),
                title="Mcs: " + str(edge.mcs) + ", SnapCoreRMSD: ",
            )
        self.vmap_output = vmap.show(os.path.join(write_out_path, "vmap.html"))
        # return self.vmap_output

    def get_protein(self) -> GenericData:
        return self.protein

    # TODO `get_` methods should mirror the python dict get method,
    # i.e. return a specified default if there's no entry
    # (general issue, not just here). Normal indexing can raise a `KeyError`.
    def get_edge_by_id(self, id: str) -> Optional[Edge]:
        """Lookup edge by identifier

        :param str id: edge hash to retrieve
        :return Optional[Edge]: Return the edge if found, else None
        """
        # handle case where the task is actually a path to a batch script
        if not isinstance(id, Edge):
            parts = id.split("/")

            for part in parts:
                for e in self.edges:
                    if part == e:
                        id = e

        match = [e for e in self.edges if e.get_edge_id() == id]
        if not match:
            return
        else:
            return match[0]

    def __repr__(self) -> str:
        return f"Icolos Perturbation Map object containing {len(self.edges)} edges and {len(self.nodes)} nodes"

    def __str__(self) -> str:
        return self.__repr__()
