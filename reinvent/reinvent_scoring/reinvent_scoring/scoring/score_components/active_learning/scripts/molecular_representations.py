import numpy as np
import pandas as pd
from tqdm import tqdm 
import concurrent.futures
from rdkit import Chem, Avalon
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.DataStructs import cDataStructs
import logging

class molecular_representations_class:
    def __init__(self):
        self.dummy = False

    def choice(self, molecularRepresentationChoice):
        if molecularRepresentationChoice.lower() == "physchem":
            return physchem()
        if molecularRepresentationChoice.lower() == "ecfp":
            return ecfp()
        if molecularRepresentationChoice.lower() == "hash_ecfp":
            return hash_ecfp()
        if molecularRepresentationChoice.lower() == "avalon":
            return avalon()
        if molecularRepresentationChoice.upper() == "MACC":
            return MACC()
    
    def fingerprints_in_parrallel(
        self, fingerprint_generator_function, iterable_of_smiles
    ):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fingerprints_iterable = executor.map(fingerprint_generator_function, iterable_of_smiles)
            return fingerprints_iterable
        
    def _rdkit_to_numpy(self, rdkit_vec, dtype):
        fp_numpy = np.zeros((0,), dtype=dtype)
        cDataStructs.ConvertToNumpyArray(rdkit_vec, fp_numpy)
        return fp_numpy

    def _catch_exception(self, function_, value):
        try:
            output_ = function_(value)
            return output_
        except Exception as e:
            logging.debug(f'Caught Exception during fingerprint generation {e}')
            return None



class physchem(molecular_representations_class):
    def __init__(self):
        self.dummy = False
        
    def calculateRepresentation(self, smiles):
        physchemObj = self.initialise_physchem()
        #descriptors_iterable = self.fingerprints_in_parrallel(physchemObj.CalcDescriptors, smiles)
        descriptors_iterable = [self._catch_exception(physchemObj.CalcDescriptors, (Chem.MolFromSmiles(x))) for x in smiles]
        descriptors = [descriptor for descriptor in descriptors_iterable]
        return descriptors
    
    def initialise_physchem(self):
        names_of_descriptors = [x[0] for x in Chem.Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(names_of_descriptors)
        return calc


class ecfp(molecular_representations_class):
    def __init__(self):
        self.dummy = False

    def calculateRepresentation(self, smiles):

        fingerprints_iterable = [self._catch_exception(self.ecfpFunction, Chem.MolFromSmiles(x)) for x in smiles]
        fingerprints = [fingerprint for fingerprint in fingerprints_iterable]
        fingerprints = np.squeeze(fingerprints)
        
        return list(fingerprints)

    def ecfpFunction(self, smile):
        descriptor = [
            Chem.AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius=3, nBits=2048)
        ]
        return descriptor

class hash_ecfp(molecular_representations_class):
    def __init__(self):
        self.dummy = False
        
    def calculateRepresentation(self, smiles):
        fingerprints_iterable = self.fingerprints_in_parrallel(self.hashedecfpFunction, smiles)
        fingerprints = [fingerprint for fingerprint in fingerprints_iterable]
        return fingerprints

    def hashedecfpFunction(self, smile):
        descriptor = [
            Chem.AllChem.GetHashedMorganFingerprint(smile, radius=3, nBits=2048, useFeatures=True)
        ]
        mol_vec = [
            self._rdkit_to_numpy(rdkit_vec=descriptor, dtype=np.int8)
        ]
        return mol_vec

class avalon(molecular_representations_class):
    def __init__(self):
        self.dummy = False
        
    def calculateRepresentation(self, mol):
        return

class MACC(molecular_representations_class):
    def __init__(self):
        self.dummy = False

    def calculateRepresentation(self, mol):
        return
             