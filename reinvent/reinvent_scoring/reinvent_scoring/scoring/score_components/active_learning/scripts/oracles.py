import subprocess
import tempfile
import os
from rdkit import Chem
import concurrent.futures
import math
import pandas as pd
import time
import json
import logging
import numpy as np 
from pathlib import Path
try:
    from maize.core.workflow import Workflow
    from maize.utilities.io import Config, NodeConfig
    from maize.steps.mai.docking.adv import AutoDockGPU, Vina
    from maize.steps.mai.molecule import Gypsum
    from maize.steps.io import Return, LoadData
    from maize.core.node import Node
    from maize.core.interface import Parameter, Input, Flag, Output, FileParameter
    from maize.utilities.io import setup_workflow
    from numpy.typing import NDArray
except: 
    pass
from typing import Any


class oracles_class:
    def __init__(
        self, confPath=False, propertyName=False, direction="positive", currentEpoch=-1
    ):
        self.dummy = False
        self.confPath = confPath
        self.propertyName = propertyName
        self.direction = direction
        self.currentEpoch = currentEpoch

    def choice(self, oracleChoice):

        if oracleChoice.lower() == "icolos":
            return icolos(
                confPath=self.confPath,
                propertyName=self.propertyName,
                direction=self.direction,
                currentEpoch=self.currentEpoch,
            )
        
    def _generate_sublists(self, smilesList, subListLen=8):
        if subListLen:
            subListLen = min(len(smilesList), subListLen)
            numLists = math.ceil((len(smilesList) / subListLen))
            masterList = []
            majorList = smilesList
            for subList in range(numLists):
                subListLen = min(subListLen, len(majorList))
                subList = majorList[:subListLen]
                majorList = majorList[subListLen:]
                masterList.append(subList)
        return masterList

    def _read_in_SDF(self, filePath, propertyName):
        supply = Chem.SDMolSupplier(f"{filePath}")
        dictionary = {}
        for mol in supply:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            property = float(mol.GetProp(propertyName))
            key = int(mol.GetProp("compound_name"))
            if self.direction.lower() == "positive":
                if key not in dictionary:
                    dictionary[key] = property
                elif dictionary[key] < property:
                    dictionary[key] = property
            else:
                if key not in dictionary:
                    dictionary[key] = property
                elif dictionary[key] > property:
                    dictionary[key] = property

        return dictionary

    def _if_no_score_add_zero(self, scoreIndexDictionary, originalSmiles):
        indexScoresDictionary = {}
        for index in range(len(originalSmiles)):
            indexScoresDictionary[index] = 0.0

        for index in scoreIndexDictionary:
            indexScoresDictionary[index] = scoreIndexDictionary[index]

        scores = list(indexScoresDictionary.values())

        return scores

class icolos(oracles_class):
    def __init__(self, confPath, propertyName, direction, currentEpoch):
        self.dummy = False
        self.confPath = confPath
        self.propertyName = propertyName
        self.direction = direction
        self.currentEpoch = currentEpoch

    def calculateScores(self, originalSmiles, loopEpoch=False):
        scores = self._calculateScoresSmiles(
            originalSmiles=originalSmiles, loopEpoch=loopEpoch
        )
        return scores

    def _calculateScoresSmiles(self, originalSmiles, loopEpoch):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ligand_path = os.path.join(tmpdirname, "input_smiles.smi")
            mol_array = [Chem.MolFromSmiles(smile) for smile in originalSmiles]
            hydrated_mols = [Chem.AddHs(mol, addCoords = True) for mol in mol_array]
            originalSmiles = [Chem.MolToSmiles(mol) for mol in hydrated_mols] 
            with open(ligand_path, "w") as f:
                for line in originalSmiles:
                    f.write(f"{line}\n")

            resultPath, icolosConfPath = self._add_writeOutBlock(
                confPath=self.confPath,
                tempoutPath=tmpdirname,
                ligand_path=ligand_path,
                loopEpoch=loopEpoch,
            )

            with open(icolosConfPath, "r") as f:
                dic = json.load(f)
            if dic["workflow"]["steps"][-1]["step_id"].upper() == "ROCS":
                prefixExecution = "module purge && module load oetoolkits"
                self.direction = "positive"
            elif dic["workflow"]["steps"][-1]["step_id"].upper() == "GLIDE":
                prefixExecution = "ml schrodinger/2022-4-GCCcore-8.2.0-js-aws && jsc local-server-start"
                self.direction = "negative"
            elif dic["workflow"]["steps"][-1]["step_id"].upper() == "ADV":
                prefixExecution = "ml AutoDock_Vina"
                self.direction = "negative"
            else:
                print(
                    f'{dic["workflow"]["steps"][-1]["step_id"].upper()} has not been implemented yet!'
                )

            environmentLoad = "source miniconda3/bin/activate icolos"

            p1 = subprocess.run(
                f"{prefixExecution} && {environmentLoad} && icolos -conf {icolosConfPath}",
                shell=True,
                capture_output=True,
                text=True,
            )

            # logging.debug(f'Printing debug information:\n{p1}')
            logging.info(
                f"command run: {prefixExecution} && {environmentLoad} && icolos -conf {icolosConfPath}"
            )

            print(
                f"This is the result path {resultPath} and it exists: {os.path.exists(resultPath)}"
            )

            # This sometimes fails, if the input number of smiles is very small and fails to generate dockable conformers
            # RDKIT will produce an SDF which contains 0 conformers, causing the mol supplier to read the file as invalid
            # Current solution is just to pass an empty dictionary, which will cause the originalSmiles passed to be scored 0
            try:
                scoreIndexDictionary = self._read_in_SDF(
                    filePath=resultPath, propertyName=self.propertyName
                )
            except OSError:
                scoreIndexDictionary = {0: 0.0}

            scores = self._if_no_score_add_zero(
                scoreIndexDictionary=scoreIndexDictionary,
                originalSmiles=originalSmiles,
            )

        return scores

    def _add_writeOutBlock(self, confPath, tempoutPath, ligand_path, loopEpoch):
        with open(confPath) as json_file:
            dic = json.load(json_file)

        newconfPath = os.path.join(tempoutPath, "icolos_conf.json")
        resultPath = os.path.join(tempoutPath, "conformers.sdf")

        if loopEpoch:
            dic["workflow"]["header"]["global_variables"][
                "step_id"
            ] = f"{self.currentEpoch}_{loopEpoch}"
        else:
            dic["workflow"]["header"]["global_variables"]["step_id"] = self.currentEpoch

        
        #TODO Temporary solution to remove the writeout block because of SCP file storage issues
        # Tagline: IO operation

        if self.direction.upper() == 'positive':
            best_bool = True
        else:
            best_bool = False 
        
        dic["workflow"]["steps"][-1:][0]["writeout"] = [
            {
                "compounds": {
                    "aggregation": {
                        "highest_is_best": best_bool,
                        "key": "docking_score",
                        "mode": "best_per_compound",
                    },
                    "category": "conformers",
                    "selected_tags": ["docking_score"],
                },
                "destination": {
                    "format": "SDF",
                    "resource": f"{resultPath}",
                    "type": "file",
                },
            }
        ]

        dic["workflow"]["steps"][0]["input"]["compounds"][0]["source"] = ligand_path

        with open(f"{tempoutPath}/icolos_conf.json", "w") as json_write:
            json.dump(dic, json_write, indent=2)

        return resultPath, newconfPath
