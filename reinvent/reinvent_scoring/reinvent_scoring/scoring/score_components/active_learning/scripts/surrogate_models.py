import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
#from xgboost import XGBRFRegressor
from sklearn.preprocessing import StandardScaler


class surrogate_models_class:
    def __init__(self):
        self.dummy = False 

    def choice(self, surrogateModelChoice, molecularRepresentationChoice):
        if surrogateModelChoice.lower() == "randomforest":
            return randomforest(molecularRepresentationChoice)
        if surrogateModelChoice.lower() == "xgboost":
            return xgboost(molecularRepresentationChoice)
        if surrogateModelChoice.upper() == "SVR":
            return SVR(molecularRepresentationChoice)
        if surrogateModelChoice.lower() == "gaussianprocess":
            return gaussianProcess(molecularRepresentationChoice)
        if surrogateModelChoice.upper() == "KNN":
            return KNN(molecularRepresentationChoice)

    def train(self, molecularRepresentations, scores):
        return "train not implemented!"

    def predict(self, trained_surrogate, molecularRepresentations):
        return "predict not implemented!"

    def _scaler(self, values):
        scaler = StandardScaler()
        self.scaler = scaler.fit(values)
        standardised_values = self.scaler.transform(values)
        return standardised_values

    def _drop_invariant_columns(self, array):
        std_array = array.std(axis=0, keepdims=True)
        self.dropped_features = np.where(std_array == 0)
        new_array = np.delete(array, np.where(std_array == 0)[1], axis=1)
        logging.debug(f'dropping invariant columns from descriptors, dropped feature array: \n {self.dropped_features}')
        return new_array

    def _drop_and_scale(self, molecularRepresentation):
        scaledRepresentation = self._scaler(molecularRepresentation)
        cleanRepresentation = self._drop_invariant_columns(
            np.array(scaledRepresentation)
        )
        return cleanRepresentation

    def _drop_columns(self, molecularRepresentation):
        array = np.delete(molecularRepresentation, self.dropped_features[1], axis=1)
        return array

    def _scale_columns(self, molecularRepresentation):
        standardised_values = self.scaler.transform(molecularRepresentation)
        return standardised_values


class randomforest(surrogate_models_class):
    def __init__(self, molecularRepresentationChoice):
        self.dummy = False
        self.molecularRepresentationChoice = molecularRepresentationChoice
        

    def train(self, molecularRepresentations, scores):
        molecularRepresentations = [x for x in molecularRepresentations]
        self.surrogate = RandomForestRegressor()
        self.surrogate.set_params(n_jobs=-1, n_estimators=200, max_depth=20)
        
        if self.molecularRepresentationChoice == 'physchem':
            molecularRepresentations = self._drop_and_scale(molecularRepresentations)

        self.trained_surrogate = self.surrogate.fit(
            molecularRepresentations, scores
        )

        return self.trained_surrogate

    def predict(self, molecularRepresentations, trained_surrogate=False):
        if len(molecularRepresentations) > 1:
            x = [np.array(x) for x in molecularRepresentations]
            molecularRepresentations = np.array(x)
        
        if self.molecularRepresentationChoice == 'physchem':
            molecularRepresentations = self._scale_columns(molecularRepresentations)
            molecularRepresentations = self._drop_columns(molecularRepresentations)

        if trained_surrogate:
            self.trained_surrogate = trained_surrogate

        predictions = self.trained_surrogate.predict(molecularRepresentations)

        confidence = self._error(molecularRepresentations)

        return predictions, confidence

    def _error(self, molecularRepresentations):
        individual_trees = self.trained_surrogate.estimators_
        subEstimates = np.array(
            [
                tree.predict(np.stack(molecularRepresentations))
                for tree in individual_trees
            ]
        )
        return np.std(subEstimates, axis=0)


class xgboost(surrogate_models_class):
    def __init__(self, molecularRepresentationChoice):
        self.dummy = False
        self.molecularRepresentationChoice = molecularRepresentationChoice

    def train(self, molecularRepresentations, scores):
        #self.surrogate = XGBRFRegressor()
        self.surrogate.set_params(n_jobs=-1, n_estimators=200, max_depth=20)
        self.molecularRepresentations = self._drop_and_scale(molecularRepresentations)
        self.trained_surrogate = self.surrogate.fit(
            self.molecularRepresentations, scores
        )

        return self.trained_surrogate

    def predict(self, molecularRepresentations, trained_surrogate=False):
        if trained_surrogate:
            self.trained_surrogate = trained_surrogate

        if len(molecularRepresentations) > 1:
            x = [np.array(x) for x in molecularRepresentations]
            molecularRepresentations = np.array(x)
        molecularRepresentations = self._scale_columns(molecularRepresentations)
        molecularRepresentations = self._drop_columns(molecularRepresentations)

        predictions = self.trained_surrogate.predict(molecularRepresentations)

        confidence = self._error(molecularRepresentations)

        return predictions, confidence

    def _error(self, molecularRepresentations):
        individual_trees = self.trained_surrogate.estimators_
        subEstimates = np.array(
            [
                tree.predict(np.stack(molecularRepresentations))
                for tree in individual_trees
            ]
        )
        return np.std(subEstimates, axis=0)


class SVR(surrogate_models_class):
    def __init__(self, molecularRepresentationChoice):
        self.dummy = False
        self.molecularRepresentationChoice = molecularRepresentationChoice

    def train(self, molecularRepresentations, scores):
        return "train not implemented!"

    def predict(self, trained_surrogate, molecularRepresentations):
        return "predict not implemented!"


class gaussianProcess(surrogate_models_class):
    def __init__(self, molecularRepresentationChoice):
        self.dummy = False
        self.molecularRepresentationChoice = molecularRepresentationChoice

    def train(self, molecularRepresentations, scores):
        kernel  = RBF()
        self.surrogate = GaussianProcessRegressor(kernel=kernel)
        self.molecularRepresentations = self._drop_and_scale(molecularRepresentations)
        self.trained_surrogate = self.surrogate.fit(
            self.molecularRepresentations, scores
        )

        return self.trained_surrogate

    def predict(self, trained_surrogate, molecularRepresentations):
        if trained_surrogate:
            self.trained_surrogate = trained_surrogate

        if len(molecularRepresentations) > 1:
            x = [np.array(x) for x in molecularRepresentations]
            molecularRepresentations = np.array(x)
        molecularRepresentations = self._scale_columns(molecularRepresentations)
        molecularRepresentations = self._drop_columns(molecularRepresentations)

        predictions, confidence = self.trained_surrogate.predict(molecularRepresentations)
        

        return predictions, confidence


class KNN(surrogate_models_class):
    def __init__(self, molecularRepresentationChoice):
        self.dummy = False
        self.molecularRepresentationChoice = molecularRepresentationChoice

    def train(self, molecularRepresentations, scores):
        return "train not implemented!"

    def predict(self, trained_surrogate, molecularRepresentations):
        return "predict not implemented!"

