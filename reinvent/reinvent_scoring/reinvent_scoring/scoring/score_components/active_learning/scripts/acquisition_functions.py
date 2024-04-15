import numpy as np
import pandas as pd
import logging
import math
import time 
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors import qed, rdMolDescriptors


class acquisition_functions_class:
    def __init__(self, relative_fraction, n_lig, num_cpus):
        self.relative_fraction = relative_fraction
        self.n_lig = n_lig
        self.num_cpus = num_cpus
        
    def sigmoid(self, predictions: list, low, high, k) -> np.array:

        def _exp(pred_val, low, high, k) -> float:
            return math.pow(10, (10 * k * (pred_val - (low + high) * 0.5) / (low - high)))
        
        transformed = [1 / (1 + _exp(pred_val, low, high, k)) for pred_val in predictions]
        
        return np.array(transformed, dtype=np.float32)

    def reverse_sigmoid(self, predictions: list, _low, _high, _k) -> np.array:

        def _reverse_sigmoid_formula(value, low, high, k) -> float:
            try:
                return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
            except:
                return 0

        if len(predictions.shape) > 1:
            transformed = [_reverse_sigmoid_formula(pred_val, _low, _high, _k) for pred in predictions for pred_val in pred]
        else:    
            transformed = [_reverse_sigmoid_formula(pred_val, _low, _high, _k) for pred_val in predictions]
        
        return np.array(transformed, dtype=np.float32)

    def double_sigmoid(self, predictions: list, _low, _high, _coef_div, _coef_si, _coef_se) -> np.array:

            def _double_sigmoid_formula(value, low, high, coef_div=100., coef_si=150., coef_se=150.):
                try:
                    A = 10 ** (coef_se * (value / coef_div))
                    B = (10 ** (coef_se * (value / coef_div)) + 10 ** (coef_se * (low / coef_div)))
                    C = (10 ** (coef_si * (value / coef_div)) / (
                            10 ** (coef_si * (value / coef_div)) + 10 ** (coef_si * (high / coef_div))))
                    return (A / B) - C
                except:
                    return 0

            transformed = [_double_sigmoid_formula(pred_val, _low, _high, _coef_div, _coef_si, _coef_se) for pred_val in
                        predictions]
            return np.array(transformed, dtype=np.float32)

    def alerts_match(self, query_mols, list_of_SMARTS):
        match = [any([mol.HasSubstructMatch(Chem.MolFromSmarts(subst)) for subst in list_of_SMARTS
                        if Chem.MolFromSmarts(subst)]) for mol in query_mols]
        reverse = [1 - m for m in match]
        return reverse

    components_to_compute = ['Molecular weight', 'simple alerts', 'QED Score', 'Number of HB-donors (Lipinksi)', 'AL']

    def _calculate_pow(self, values: list, weight: np.float32) -> np.array:
        y = [math.pow(value, weight) for value in values]
        return np.array(y, dtype=np.float32)
        

    def get_weight_score(self, mol_objects: list) -> np.array:
        weights = np.array([Descriptors.MolWt(mol) for mol in mol_objects])
        return self.double_sigmoid(weights, _low = 200, _high=550, _coef_div=550, _coef_si=20, _coef_se=20)

    def get_qed_score(self, mol_objects: list) -> np.array:
        return np.array([qed(mol) for mol in mol_objects])

    def get_hbd_score(self, mol_objects: list) -> np.array:
        hbd_scores = np.array([rdMolDescriptors.CalcNumHBD(mol) for mol in mol_objects])
        return self.reverse_sigmoid(hbd_scores, _low=2, _high=6, _k=0.5)

    def get_alerts_score(self, mol_objects: list) -> np.array:
        smarts = [ "[*;r7]", "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]", "[*;r13]", "[*;r14]", "[*;r15]", "[*;r16]", "[*;r17]", "[#8][#8]", "[#6;+]", "[#16][#16]", "[#7;!n][S;!$(S(=O)=O)]", "[#7;!n][#7;!n]", "C#C", "C=N", "C(=[O,S])[O,S]"]
        return np.array(self.alerts_match(mol_objects, smarts))

    def get_al_score_rocs(self, oracle_scores: list) -> np.array:
        return self.sigmoid(oracle_scores, low=0.3, high=0.65, k=0.3)

    def get_al_score_adv(self, oracle_scores: list) -> np.array:
        return self.reverse_sigmoid(oracle_scores, _low=-13.5, _high=-6, _k=0.2)

    def get_total_score(self, df: pd.DataFrame, adv: bool = False):
        df['weight_score'] = self.get_weight_score(df["rdkit_mol_objects"].to_numpy())
        df['hbd_score'] = self.get_hbd_score(df["rdkit_mol_objects"].to_numpy())
        df['alert_score'] = self.get_alerts_score(df["rdkit_mol_objects"].to_numpy())
        df['qed_score'] = self.get_qed_score(df["rdkit_mol_objects"].to_numpy())
        if adv:
            df['al_score'] = self.get_al_score_adv(df['af_metric'].to_numpy())
        else:
            df['al_score'] = self.get_al_score_rocs(df['af_metric'].to_numpy())
        
        components=['weight_score', 'hbd_score', 'al_score', 'qed_score']
        penalty_components = ['alert_score']

        product = np.full(len(df), 1, dtype=np.float32)
        weight = 1 / len(components)
        
        for component in components:
            array = df[component].to_numpy()
            comp_pow = self._calculate_pow(array, weight)
            product = product * comp_pow

        for component in penalty_components:
            array = df[component].to_numpy()
            penalty = [0.0 if x == 0 else x for x in array]

        product = product * penalty     
        return product 
    
    def compute_non_penalty_d(self, mol_array):
        mol_weight_array =  self.get_weight_score(mol_array)
        hbd_array = self.get_hbd_score(mol_array)
        qed_array = self.get_qed_score(mol_array)
        product = np.product([mol_weight_array, hbd_array, qed_array], axis=0)
        return product.reshape(-1, 1)
    
    def compute_non_penalty_nd(self, component_scores, num_samples, deviation, adv = False):
        
        sampled_distribution_array = np.random.multivariate_normal(mean=component_scores, cov=np.diag(abs(deviation)), size=num_samples).T
        
        if adv:
            logging.info('Computing ADV AL Score')
            transformed_distribution_array = np.array([self.get_al_score_adv(array) for array in sampled_distribution_array])
        else:  
            logging.info('Computing ROCS AL Score')
            transformed_distribution_array = np.array([self.get_al_score_rocs(array) for array in sampled_distribution_array])

        return transformed_distribution_array

    def compute_penalty(self, mol_array):
        penalty_array = self.get_alerts_score(mol_array)
        return penalty_array.reshape(-1, 1)
    
    def compute_transformed_scores(self, batch, num_samples, adv = False):
        mol = batch["rdkit_mol_objects"].to_numpy()
        non_penalty_product_d = self.compute_non_penalty_d(mol)
        
        if 'confidence' in batch.columns: 
            non_penalty_product_nd = self.compute_non_penalty_nd(batch['predictions'].to_numpy(), deviation = batch['confidence'].to_numpy(), num_samples=num_samples, adv=adv)
        else:     
            non_penalty_product_nd = self.compute_non_penalty_nd(batch['predictions'].to_numpy(), deviation = np.repeat(np.array(0), repeats=len(batch)), num_samples=num_samples, adv=adv)
        
        penalty_product = self.compute_penalty(mol)

        #Multiplying the non penalty component, with the penalty component seperately.
        non_penalty_product = np.multiply(non_penalty_product_d, non_penalty_product_nd)
        total_product = np.multiply(non_penalty_product, penalty_product)
        
        #Compute the geometric mean of the transformed values to get the distribution of the transformed score, as well as the central tendency, and deviation.
        geometric_mean_transformed_values = np.power(total_product, 1/4)
        return geometric_mean_transformed_values

    def montecarlo_sampling(self, batch, adv):
        transformed_scores = self.compute_transformed_scores(batch, num_samples=1000, adv = adv)
        mean_prediction = np.mean(transformed_scores, axis=1)
        mean_std = np.std(transformed_scores, axis=1)
        
        batch['mean_transform'] = mean_prediction
        batch['mean_transform_dev'] = mean_std

        return batch 

    def choice(self, acquisitionFunctionChoice):
        if acquisitionFunctionChoice.lower() == "random":
            return random(
                relative_fraction=self.relative_fraction, num_cpus=self.num_cpus
            )

        if acquisitionFunctionChoice.upper() == "UCB":
            return UCB(
                relative_fraction=self.relative_fraction,
                n_lig=self.n_lig,
                num_cpus=self.num_cpus,
            )

        if acquisitionFunctionChoice.lower() == "greedy":
            return greedy(
                relative_fraction=self.relative_fraction,
                n_lig=self.n_lig,
                num_cpus=self.num_cpus,
            )

        if acquisitionFunctionChoice.lower() == "uncertain":
            return uncertain(
                relative_fraction=self.relative_fraction,
                n_lig=self.n_lig,
                num_cpus=self.num_cpus,
            )

    def extract_frames(self, compoundDataFrame, acquisitionBatchSize, r_hits, n_hits):
        
        best_prediction = compoundDataFrame.iloc[0].loc["af_metric"]
        cutoff_point = best_prediction * self.relative_fraction

        if np.float32(self.n_lig) != 0.0 and n_hits > acquisitionBatchSize:
            acquisitionBatchSize = n_hits
            logging.debug(
                f"There are {n_hits} df predicted to be better than the native ligand\nThere are {r_hits} df found inside the defined ratio of the best prediction {best_prediction} / {self.relative_fraction} = {np.round(cutoff_point,2)}"
            )
        if np.float32(self.relative_fraction) != 0.0 and np.float32(self.n_lig) == 0.0 or r_hits > n_hits :
            acquisitionBatchSize = r_hits

        if int(self.num_cpus) != 0:
            acquisitionBatchSize = (
                math.ceil(acquisitionBatchSize / self.num_cpus) * self.num_cpus
            )

        toAcquireDataframe, dontAcquireDataframe = self._split_into_correct_batch(
            compoundDataFrame, acquisitionBatchSize
        )

        return toAcquireDataframe, dontAcquireDataframe

    def partition(
        self,
        compoundDataFrame,
        acquirebatchSize,
    ):
        return f"{self.acquisitionFunctionChoice} not implemented yet"

    def _split_into_correct_batch(
        self,
        compoundDataFrame: pd.DataFrame,
        acquisitionBatchSize: int,
    ):

        toAcquireDataframe, dontAcquireDataframe = (
            compoundDataFrame[:acquisitionBatchSize],
            compoundDataFrame[acquisitionBatchSize:],
        )
        return toAcquireDataframe, dontAcquireDataframe

    def calc_qed(self, mol_objects):
        qed_scores = [qed(mol) for mol in mol_objects]
        return qed_scores 

class random(acquisition_functions_class):
    def __init__(self, relative_fraction, num_cpus):
        self.num_cpus = num_cpus

    def partition(
        self,
        compoundDataFrame,
        acquisitionBatchSize,
        direction,
        mpo_acquisition,
        ucb_beta_value, 
    ):

        compoundDataFrame = compoundDataFrame.sample(frac=1)

        toAcquireDataframe, dontAcquireDataframe = self._split_into_correct_batch(
            compoundDataFrame, acquisitionBatchSize
        )

        if int(self.num_cpus) != 0:
            acquisitionBatchSize = (
                math.ceil(acquisitionBatchSize / self.num_cpus) * self.num_cpus
            )

        return toAcquireDataframe, dontAcquireDataframe


class UCB(acquisition_functions_class):
    def __init__(self, relative_fraction, n_lig, num_cpus):
        self.relative_fraction = relative_fraction
        self.n_lig = n_lig
        self.num_cpus = num_cpus

    def partition(
        self,
        compoundDataFrame,
        acquisitionBatchSize,
        direction,
        mpo_acquisition,
        ucb_beta_value
    ):
        
        ucb_weight = ucb_beta_value 
        logging.info(f'UCB Weight : {ucb_weight}')

        if direction == "positive":
            compoundDataFrame["af_metric"] = compoundDataFrame["predictions"] + (
                ucb_weight * compoundDataFrame["confidence"]
            )
            adv = False
            if mpo_acquisition:
                compoundDataFrame["af_metric"] = self.get_total_score(df=compoundDataFrame, adv=adv)
                compoundDataFrame = self.montecarlo_sampling(compoundDataFrame, adv=adv)
                compoundDataFrame["af_metric"] = compoundDataFrame['mean_transform'] + (ucb_weight * compoundDataFrame['mean_transform_dev'])
            
            compoundDataFrame = compoundDataFrame.sort_values(
                "af_metric", ascending=False
            )

            r_hits = len(
                compoundDataFrame.loc[
                    compoundDataFrame["af_metric"]
                    > (
                        compoundDataFrame.iloc[0].loc["af_metric"]
                        * self.relative_fraction
                    )
                ]
            )
            n_hits = len(
                compoundDataFrame.loc[compoundDataFrame["af_metric"] > self.n_lig]
            )
        elif direction == 'negative':
            compoundDataFrame["af_metric"] = compoundDataFrame["predictions"] - (
                ucb_weight * compoundDataFrame["confidence"]
            )
            adv = True
            if mpo_acquisition:
                #compoundDataFrame["af_metric"] = self.get_total_score(df=compoundDataFrame, adv=adv)
                compoundDataFrame = self.montecarlo_sampling(compoundDataFrame, adv=adv)
                compoundDataFrame["af_metric"] = compoundDataFrame['mean_transform'] - (ucb_weight * compoundDataFrame['mean_transform_dev'])

            compoundDataFrame = compoundDataFrame.sort_values(
                "af_metric", ascending=True
            )
            r_hits = len(
                compoundDataFrame.loc[
                    compoundDataFrame["af_metric"]
                    < (
                        compoundDataFrame.iloc[0].loc["af_metric"]
                        * self.relative_fraction
                    )
                ]
            )
            n_hits = len(
                compoundDataFrame.loc[compoundDataFrame["af_metric"] < self.n_lig]
            )

        if mpo_acquisition:
            if 'af_metric' in compoundDataFrame.columns:
                compoundDataFrame = compoundDataFrame.sort_values(by="af_metric", ascending=False)
                logging.info(msg = f'Top five compounds from acquisition metric {compoundDataFrame["af_metric"][0:5]}') 
        
        toAcquireDataframe, dontAcquireDataframe = self.extract_frames(
            compoundDataFrame, acquisitionBatchSize, r_hits, n_hits
        )

        return toAcquireDataframe, dontAcquireDataframe


class greedy(acquisition_functions_class):
    def __init__(self, relative_fraction, n_lig, num_cpus):
        self.relative_fraction = relative_fraction
        self.n_lig = n_lig
        self.num_cpus = num_cpus

    def partition(
        self,
        compoundDataFrame,
        acquisitionBatchSize,
        direction,
        mpo_acquisition,
        ucb_beta_value, 
    ):
        compoundDataFrame["af_metric"] = compoundDataFrame["predictions"].to_numpy()
        if mpo_acquisition:
                compoundDataFrame["af_metric"] = compoundDataFrame["af_metric"] * compoundDataFrame["qed"]

        if direction == "positive":
            compoundDataFrame = compoundDataFrame.sort_values(
                "af_metric", ascending=False
            )
            n_hits = len(
                compoundDataFrame.loc[compoundDataFrame["af_metric"] > self.n_lig]
            )
            r_hits = len(
                compoundDataFrame.loc[
                    compoundDataFrame["af_metric"]
                    > (
                        compoundDataFrame.iloc[0].loc["af_metric"]
                        * self.relative_fraction
                    )
                ]
            )
        else:
            if mpo_acquisition:
                compoundDataFrame["af_metric"] = compoundDataFrame["af_metric"] * compoundDataFrame["qed"]

            compoundDataFrame = compoundDataFrame.sort_values(
                "af_metric", ascending=True
            )
            n_hits = len(
                compoundDataFrame.loc[compoundDataFrame["af_metric"] < self.n_lig]
            )
            r_hits = len(
                compoundDataFrame.loc[
                    compoundDataFrame["af_metric"]
                    < (
                        compoundDataFrame.iloc[0].loc["af_metric"]
                        * self.relative_fraction
                    )
                ]
            )
        
        toAcquireDataframe, dontAcquireDataframe = self.extract_frames(
            compoundDataFrame, acquisitionBatchSize, r_hits, n_hits
        )

        return toAcquireDataframe, dontAcquireDataframe

class uncertain(acquisition_functions_class):
    def __init__(self, relative_fraction, n_lig, num_cpus):
        self.relative_fraction = relative_fraction
        self.n_lig = n_lig
        self.num_cpus = num_cpus

    def partition(
        self,
        compoundDataFrame,
        acquisitionBatchSize,
        direction,
        mpo_acquisition,
        ucb_beta_value, 
    ):
        compoundDataFrame["af_metric"] = compoundDataFrame["confidence"].to_numpy()

        if direction == "positive":
            compoundDataFrame = compoundDataFrame.sort_values(
                "af_metric", ascending=False
            )
            n_hits = len(
                compoundDataFrame.loc[compoundDataFrame["af_metric"] > self.n_lig]
            )
            r_hits = len(
                compoundDataFrame.loc[
                    compoundDataFrame["af_metric"]
                    > (
                        compoundDataFrame.iloc[0].loc["af_metric"]
                        * self.relative_fraction
                    )
                ]
            )
        else:
            compoundDataFrame = compoundDataFrame.sort_values(
                "af_metric", ascending=True
            )
            n_hits = len(
                compoundDataFrame.loc[compoundDataFrame["af_metric"] < self.n_lig]
            )
            r_hits = len(
                compoundDataFrame.loc[
                    compoundDataFrame["af_metric"]
                    < (
                        compoundDataFrame.iloc[0].loc["af_metric"]
                        * self.relative_fraction
                    )
                ]
            )

        toAcquireDataframe, dontAcquireDataframe = self.extract_frames(
            compoundDataFrame, acquisitionBatchSize, r_hits, n_hits
        )

        return toAcquireDataframe, dontAcquireDataframe
