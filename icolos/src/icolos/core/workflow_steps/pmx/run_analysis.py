""" Run analysis - Analyse the output of the FEP simulations """

import glob
import os

import numpy as np
import pandas as pd  # type: ignore
from pydantic import BaseModel

from icolos.core.containers.perturbation_map import Edge
from icolos.core.workflow_steps.pmx.base import StepPMXBase
from icolos.core.workflow_steps.step import _LE
from icolos.utils.execute_external.pmx import PMXExecutor
from icolos.utils.general.parallelization import SubtaskContainer


class StepPMXRunAnalysis(StepPMXBase, BaseModel):
    """
    Analyses map, returns both a summary and a full results dataframe, written to top level of work_dir, and attaches properties to the compound

    Requires
    --------
    :work_dir dir: Working directory containing output of previous steps, optional
    :compounds perturbation_map: Previously constructed perturbation map

    Settings
    --------
    :additional fail_score: score to use for failed edges, default = 100
    :additional exp_results: CSV file of experimental results to compare to, optional

    """

    results_summary: pd.DataFrame = None
    results_all: pd.DataFrame = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        self._initialize_backend(executor=PMXExecutor)
        self.results_summary = pd.DataFrame()
        self.results_all = pd.DataFrame()

    def execute(self):
        self.execution.parallelization.max_length_sublists = 1
        self._subtask_container = SubtaskContainer(
            max_tries=self.execution.failure_policy.n_tries
        )
        self._subtask_container.load_data(self.edge_ids)
        self._execute_pmx_step_parallel(
            run_func=self.run_analysis,
            step_id="pmx_run_analysis",
            result_checker=self._check_result,
        )
        self.analysis_summary(self.edges)
        # reattach compounds from perturbation map to step for writeout
        # REINVENT expects the same number of compounds back, if they failed to dock, they need to report a 0.00 score

        # discard the hub compound
        self.data.compounds = self.get_perturbation_map().compounds[1:]

        #     output_conf = edge.node_to.conformer
        #     enum: Enumeration = output_conf.get_enumeration_object()
        #     comp: Compound = enum.get_compound_object()
        #     print(enum, comp)
        #     self.data.compounds[comp.get_compound_number()].get_enumerations()[
        #         enum.get_enumeration_id()
        #     ].get_conformers()[0].set_molecule(output_conf.get_molecule())
        #     # print(comp, enum, conf)
        #     # match the output conformer to the compounds attached to the step from docking
        #     # self.data.compounds[int(comp)].get_enumerations[int(enum)].get_conformers[
        #     #     int(conf)
        #     # ] = output_conf
        #     self._logger.log(
        #         f"attached conf to step data for {output_conf.get_index_string()}",
        #         _LE.DEBUG,
        #     )

        # Edges that failed will have 0.00 attached, compounds that failed to dock and were never part of the map will get caught by the writeout method and set to 0.00

        # # the hub compound will not have data attached, this will be pruned here
        # all_confs = comp.unroll_conformers()
        # attached_prop = False
        # # check all conformers attached to compound, if ddG tag was attached, append to step compounds
        # if any(["ddG" in conf.get_molecule().GetPropNames() for conf in all_confs]):

        #     self.data.compounds.append(comp)

        # # self._logger.log(f"Failed to attach compound {comp.get_index_string()}, error was {e}", _LE.WARNING)
        # continue

    def _run_analysis_script(self, analysispath: str, stateApath: str, stateBpath: str):
        """
        Run PMX analyse.

        :param analysispath: output path for the analysis
        :param stateApath: path to the endpoint A simulation
        :param stateBpath: path to the endpoint B simulation

        """
        fA = " ".join(glob.glob("{0}/*xvg".format(stateApath)))
        fB = " ".join(glob.glob("{0}/*xvg".format(stateBpath)))
        oA = "integ0.dat"
        oB = "integ1.dat"
        wplot = "wplot.png"
        o = "results.txt"
        # TODO: at the moment we ignore flags from the command line
        # args = " ".join(self.settings.arguments.flags)

        cmd = "$PMX analyse"
        args = [
            "--quiet",
            "-fA",
            fA,
            "-fB",
            fB,
            "-o",
            o,
            "-oA",
            oA,
            "-oB",
            oB,
            "-w",
            wplot,
            "-t",
            298,
            "-b",
            100,
        ]
        self._backend_executor.execute(
            command=cmd, arguments=args, location=analysispath, check=False
        )

    def _read_neq_results(self, fname: str) -> list[str]:
        """
        Reads the analysis output as generated from PMX.

        :param fname: Analysis report filename

        """
        try:
            with open(fname, "r") as fp:
                lines = fp.readlines()
        except FileNotFoundError:
            return
        out = []
        for l in lines:
            l = l.rstrip()
            foo = l.split()
            if "BAR: dG" in l:
                out.append(float(foo[-2]))
            elif "BAR: Std Err (bootstrap)" in l:
                out.append(float(foo[-2]))
            elif "BAR: Std Err (analytical)" in l:
                out.append(float(foo[-2]))
            elif "0->1" in l:
                out.append(int(foo[-1]))
            elif "1->0" in l:
                out.append(int(foo[-1]))
        return out

    def _fill_resultsAll(self, res: list[str], edge: str, wp: str, r: str):
        """
        Fill dataframe with parsed results.

        :param res: result vector read from PMX
        :param edge: edge ID corresponding to result
        :param wp: thermodynamic state
        :param r: replica

        """
        try:
            rowName = "{0}_{1}_{2}".format(edge, wp, r)
            self.results_all.loc[rowName, "val"] = res[2]
            self.results_all.loc[rowName, "err_analyt"] = res[3]
            self.results_all.loc[rowName, "err_boot"] = res[4]
            self.results_all.loc[rowName, "framesA"] = res[0]
            self.results_all.loc[rowName, "framesB"] = res[1]
        except IndexError:
            self._logger.log(
                f"Index Error encountered whilst parsing results to summary file for job {edge}/{wp}/{r}",
                _LE.WARNING,
            )

    def _summarize_results(self, edges: list[Edge]):
        """
        Summarizes perturbation map results.

        :param edges: list of edges

        """
        fail_score = self._get_additional_setting("fail_score", default=100.0)
        bootnum = 1000
        for edge in edges:
            try:
                for wp in self.therm_cycle_branches:
                    dg = []
                    erra = []
                    errb = []
                    distra = []
                    distrb = []
                    for r in range(1, self.get_perturbation_map().replicas + 1):
                        rowName = "{0}_{1}_{2}".format(edge.get_edge_id(), wp, r)
                        dg.append(self.results_all.loc[rowName, "val"])
                        erra.append(self.results_all.loc[rowName, "err_analyt"])
                        errb.append(self.results_all.loc[rowName, "err_boot"])
                        distra.append(
                            np.random.normal(
                                self.results_all.loc[rowName, "val"],
                                self.results_all.loc[rowName, "err_analyt"],
                                size=bootnum,
                            )
                        )
                        distrb.append(
                            np.random.normal(
                                self.results_all.loc[rowName, "val"],
                                self.results_all.loc[rowName, "err_boot"],
                                size=bootnum,
                            )
                        )

                    rowName = "{0}_{1}".format(edge.get_edge_id(), wp)
                    distra = np.array(distra).flatten()
                    distrb = np.array(distrb).flatten()

                    if self.get_perturbation_map().replicas == 1:
                        self.results_all.loc[rowName, "val"] = dg[0]
                        self.results_all.loc[rowName, "err_analyt"] = erra[0]
                        self.results_all.loc[rowName, "err_boot"] = errb[0]
                    else:
                        self.results_all.loc[rowName, "val"] = np.mean(dg)
                        self.results_all.loc[rowName, "err_analyt"] = np.sqrt(
                            np.var(distra) / float(self.get_perturbation_map().replicas)
                        )
                        self.results_all.loc[rowName, "err_boot"] = np.sqrt(
                            np.var(distrb) / float(self.get_perturbation_map().replicas)
                        )

                # also collect self.results_summary
                rowNameWater = "{0}_{1}".format(edge.get_edge_id(), "unbound")
                rowNameProtein = "{0}_{1}".format(edge.get_edge_id(), "bound")
                dg = (
                    self.results_all.loc[rowNameProtein, "val"]
                    - self.results_all.loc[rowNameWater, "val"]
                )
                edge.ddG = dg
                try:
                    edge.node_to.get_conformer().get_molecule().SetProp("ddG", str(dg))
                    self._logger.log(
                        f"Attached score {dg} to conformer {edge.node_to.get_conformer().get_index_string()}",
                        _LE.DEBUG,
                    )
                except AttributeError as e:
                    self._logger.log(
                        f"Could not attach score to mol for edge {edge.get_edge_id()}, defaulting to {fail_score}",
                        _LE.WARNING,
                    )
                    edge.node_to.get_conformer().get_molecule().SetProp(
                        "ddG", str(fail_score)
                    )

                erra = np.sqrt(
                    np.power(self.results_all.loc[rowNameProtein, "err_analyt"], 2.0)
                    + np.power(self.results_all.loc[rowNameWater, "err_analyt"], 2.0)
                )
                edge.ddG_err = erra
                errb = np.sqrt(
                    np.power(self.results_all.loc[rowNameProtein, "err_boot"], 2.0)
                    + np.power(self.results_all.loc[rowNameWater, "err_boot"], 2.0)
                )
                rowName = edge.get_edge_id()

                self.results_summary.loc[rowName, "lig1"] = edge.get_edge_id().split(
                    "_"
                )[0]
                self.results_summary.loc[rowName, "lig2"] = edge.get_edge_id().split(
                    "_"
                )[1]
                self.results_summary.loc[rowName, "val"] = dg
                self.results_summary.loc[rowName, "err_analyt"] = erra
                self.results_summary.loc[rowName, "err_boot"] = errb

            except KeyError as e:
                print(f"Error in generating summary, error was {e}")

    def analysis_summary(self, edges: list[Edge]):
        """
        Create a full analysis summary.

        :param edges: list of edges

        """
        edge_ids = [e.get_edge_id() for e in edges]
        for edge in edge_ids:
            for r in range(1, self.get_perturbation_map().replicas + 1):
                for wp in self.therm_cycle_branches:
                    analysispath = "{0}/analyse{1}".format(
                        self._get_specific_path(
                            work_path=self.work_dir, edge=edge, therm_cycle=wp
                        ),
                        r,
                    )
                    resultsfile = "{0}/results.txt".format(analysispath)
                    res = self._read_neq_results(resultsfile)
                    if res is not None:
                        self._fill_resultsAll(res, edge, wp, r)

        # the values have been collected now
        # let's calculate ddGs
        self._summarize_results(edges)
        # compare with experimental results automatically if provided
        try:
            if "exp_results" in self.settings.additional.keys() and os.path.isfile(
                self.settings.additional["exp_results"]
            ):
                exp_data = pd.read_csv(
                    self.settings.additional["exp_results"],
                    converters={"Ligand": lambda x: str(x).split(".")[0]},
                )
                # compute the experimental ddG and append to resultsSummary
                node_data = self.get_perturbation_map().node_df
                self.results_summary["exp_ddG"] = self.results_summary.apply(
                    lambda x: np.array(
                        self.compute_exp_ddG(
                            x["lig1"], x["lig2"], node_data=node_data, exp_data=exp_data
                        )
                    ),
                    axis=1,
                )
        except Exception as e:
            self._logger.log(
                f"Failed to compute experimental results, error was: {e}", _LE.WARNING
            )
        # final write to disk
        self.results_summary.to_csv(os.path.join(self.work_dir, "resultsSummary.csv"))
        self.results_all.to_csv(os.path.join(self.work_dir, "resultsAll.csv"))

    def compute_exp_ddG(
        self, lig1: str, lig2: str, node_data: pd.DataFrame, exp_data: pd.DataFrame
    ) -> float:
        """
        Compute the ddG between two ligands from experimental data.

        :param lig1: name of ligand 1
        :param lig2: name of ligand 2
        :param node_data: dataframe with per-node data
        :param exp_data: dataframe with experimental data

        """
        lig1_id = (
            node_data.loc[node_data["hash_id"] == lig1]["node_id"]
            .to_list()[0]
            .replace(" ", "")
        )
        lig2_id = (
            node_data.loc[node_data["hash_id"] == lig2]["node_id"]
            .to_list()[0]
            .replace(" ", "")
        )
        lig1_dG = float(
            exp_data.loc[exp_data["Ligand"] == lig1_id]["Exp. ΔG"].tolist()[0]
        )
        lig2_dG = float(
            exp_data.loc[exp_data["Ligand"] == lig2_id]["Exp. ΔG"].tolist()[0]
        )
        return lig2_dG - lig1_dG

    def run_analysis(self, jobs: list[str]):
        """
        Run the full analysis.

        :param jobs: jobs in the form of edges

        """
        for idx, edge in enumerate(jobs):

            for r in range(1, self.get_perturbation_map().replicas + 1):

                # ligand
                wp = "unbound"
                analysispath = "{0}/analyse{1}".format(
                    self._get_specific_path(work_path=self.work_dir, edge=edge, therm_cycle=wp),
                    r,
                )
                os.makedirs(analysispath, exist_ok=True)
                stateApath = self._get_specific_path(
                    work_path=self.work_dir,
                    edge=edge,
                    therm_cycle=wp,
                    state="stateA",
                    replica=r,
                    sim="transitions",
                )
                stateBpath = self._get_specific_path(
                    work_path=self.work_dir,
                    edge=edge,
                    therm_cycle=wp,
                    state="stateB",
                    replica=r,
                    sim="transitions",
                )
                self._run_analysis_script(analysispath, stateApath, stateBpath)
                # protein
                wp = "bound"
                analysispath = "{0}/analyse{1}".format(
                    self._get_specific_path(work_path=self.work_dir, edge=edge, therm_cycle=wp),
                    r,
                )
                os.makedirs(analysispath, exist_ok=True)
                stateApath = self._get_specific_path(
                    work_path=self.work_dir,
                    edge=edge,
                    therm_cycle=wp,
                    state="stateA",
                    replica=r,
                    sim="transitions",
                )
                stateBpath = self._get_specific_path(
                    work_path=self.work_dir,
                    edge=edge,
                    therm_cycle=wp,
                    state="stateB",
                    replica=r,
                    sim="transitions",
                )
                self._run_analysis_script(analysispath, stateApath, stateBpath)

    def _check_result(self, batch: list[list[str]]) -> list[list[bool]]:
        """
        Look in each hybridStrTop dir and check the output pdb files exist for the edges

        :param batch: nested list of edges
        :returns results: nested list of successes

        """
        output_files = ["integ0.dat", "integ1.dat", "results.txt", "wplot.png"]
        analyse_folders = [
            f"analyse{i}" for i in range(1, self.get_perturbation_map().replicas + 1)
        ]
        results = []
        for subjob in batch:
            subjob_results = []
            for job in subjob:
                subjob_results.append(
                    all(
                        os.path.isfile(
                            os.path.join(self.work_dir, job, branch, folder, f))
                        for f in output_files
                        for branch in self.therm_cycle_branches
                        for folder in analyse_folders))
            results.append(subjob_results)
        return results
