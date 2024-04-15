import numpy as np
import pandas as pd
import os
import time
import pickle
from ast import literal_eval
from rdkit import Chem
from tqdm import tqdm
import logging
import datetime

from reinvent_scoring.scoring.score_components.active_learning.scripts.molecular_representations import (
    molecular_representations_class,
)
from reinvent_scoring.scoring.score_components.active_learning.scripts.oracles import (
    oracles_class,
)
from reinvent_scoring.scoring.score_components.active_learning.scripts.surrogate_models import (
    surrogate_models_class,
)
from reinvent_scoring.scoring.score_components.active_learning.scripts.acquisition_functions import (
    acquisition_functions_class,
)


def run_AL(
    currentEpoch: int,
    warmup: int,
    originalSmiles: str,
    trainingPoolEpochStart: int,
    trainingPoolLimit: int,
    trainingPoolMaxCompounds: int,
    molecularRepresentationChoice: str,
    oracleChoice: str,
    confPath: os.path,
    propertyName: str,
    surrogateModelChoice: str,
    acquisitionFunctionChoice: str,
    acquisitionBatchSize: int,
    pathToOutDir: os.path,
    direction: str,
    virtualLibraryLoops: int,
    virtualLibraryAcquisitionBatchSize: int,
    loopCondition: int,
    predicted_weights: np.float32,
    noise_level: np.float32,
    confidence_cutoff: np.float32,
    relative_fraction: np.float32,
    drop_duplicate_smiles: bool,
    n_lig: np.float32,
    num_cpus: int,
    max_docking_calls: int,
    subsample: bool, 
    mpo_acquisition: bool,
    ucb_beta_value: np.float32, 
):
    (
        reinventBatchFile,
        pathToOracleAcquiredCompoundsDir,
        pathToSurrogateAcquiredCompoundsDir,
        surrogateModelCheckPoints,
        trainedSurrogateModelCheckPoints,
    ) = _create_directories(pathToOutDir=pathToOutDir)

    (
        originalSmiles_dataframe,
        molecularRepresentationObject,
        surrogateModelObject,
        acquisitionFunctionObject,
        oracleObject,
    ) = _initialise_arguments(
        currentEpoch,
        originalSmiles,
        molecularRepresentationChoice,
        oracleChoice,
        confPath,
        propertyName,
        surrogateModelChoice,
        acquisitionFunctionChoice,
        direction,
        relative_fraction,
        reinventBatchFile,
        n_lig,
        num_cpus,
    )

    currentEpoch = currentEpoch + 1
    # saving a copy of the smiles order
    smiles_order_dictionary = {k: i for i, k in enumerate(originalSmiles)}

    # Initialize the counter to 0 if the file doesn't exist
    docking_counter_file_path = os.path.join(pathToOutDir, "docking_counter.txt")
    if not os.path.exists(docking_counter_file_path):
        with open(docking_counter_file_path, "w") as f:
            f.write("0")

    # Open the file and read the current value of the counter
    with open(docking_counter_file_path, "r") as f:
        docking_counter = int(f.read().strip())

    if docking_counter > trainingPoolMaxCompounds and subsample:
        subsample = True 
    else: 
        subsample = False 

    # TODO Hard coded cut-off for script, this is a temporary measure as using a dynamic cutoff
    # we do not know when the correct number of docking steps for comparison will have been performed.
    if docking_counter > max_docking_calls:
        raise ValueError(
            f"Docking Calls are greater than requested amount: {max_docking_calls}"
        )

    virtualScreenCondition = _virtual_screen_condition(
        currentEpoch, warmup, loopCondition
    )

    if currentEpoch <= warmup:
        start = time.perf_counter()
        logging.info(
            f"Starting passthrough REINVENT, acquiring all smiles, BatchSize: {len(originalSmiles)}"
        )

        (
            normal_reinvent_acquired,
            normal_reinvent_non_acquired,
        ) = _random_partition(
            originalSmiles_dataframe,
            acquisitionBatchSize=acquisitionBatchSize,
            relative_fraction=relative_fraction,
            num_cpus=num_cpus,
            ucb_beta_value=ucb_beta_value
        )

        oracle_acquired_data = normal_reinvent(
            smiles_dataframe=normal_reinvent_acquired,
            oracleObject=oracleObject,
        )

        stop = time.perf_counter()
        logging.info(
            f"Finished passthrough REINVENT in {np.round(stop-start, 2)}(s), Mean score: {np.round(np.mean(oracle_acquired_data['scores'].to_numpy()), 2)}, Number of Scores: {len(oracle_acquired_data)}"
        )

    else:
        smileWithMol, failed_smiles = _validate_smiles(originalSmiles=originalSmiles)

        if drop_duplicate_smiles:
            smileWithMol, non_unique_smileWithMol = _drop_duplicates(smileWithMol)
        else:
            non_unique_smileWithMol = None

    if currentEpoch > warmup and not virtualScreenCondition:
        start = time.perf_counter()
        logging.info(
            f"Starting active learning REINVENT, BatchSize: {acquisitionBatchSize} from {len(originalSmiles)}"
        )

        oracle_acquired_data, surrogate_acquired_data = reinvent_with_active_learning(
            currentEpoch=currentEpoch,
            smileWithMol=smileWithMol,
            trainingPoolEpochStart=trainingPoolEpochStart,
            trainingPoolLimit=trainingPoolLimit,
            trainingPoolMaxCompounds=trainingPoolMaxCompounds,
            pathToSurrogateAcquiredCompoundsDir=pathToSurrogateAcquiredCompoundsDir,
            pathToOracleAcquiredCompoundsDir=pathToOracleAcquiredCompoundsDir,
            surrogateModelCheckPoints=surrogateModelCheckPoints,
            trainedSurrogateModelCheckPoints=trainedSurrogateModelCheckPoints,
            molecularRepresentationObject=molecularRepresentationObject,
            oracleObject=oracleObject,
            surrogateModelObject=surrogateModelObject,
            acquisitionFunctionObject=acquisitionFunctionObject,
            acquisitionBatchSize=acquisitionBatchSize,
            direction=direction,
            mpo_acquisition=mpo_acquisition, 
            ucb_beta_value = ucb_beta_value
        )

        stop = time.perf_counter()
        logging.info(
            f'Finished active learning REINVENT in {np.round(stop-start, 2)}(s), Acquired scores: {oracle_acquired_data.agg({"scores": ["min", "max", "mean", "median"]})}'
        )

    if currentEpoch > warmup and virtualScreenCondition:
        start = time.perf_counter()
        logging.info(
            f"Starting active learning REINVENT, BatchSize: {virtualLibraryAcquisitionBatchSize} * {virtualLibraryLoops} loops from {len(originalSmiles)}"
        )

        oracle_acquired_data, surrogate_acquired_data = REINVENT_virtual_library_screen(
            pathToOutDir=pathToOutDir,
            smileWithMol=smileWithMol,
            oracleChoice=oracleChoice,
            confPath=confPath,
            propertyName=propertyName,
            molecularRepresentationChoice=molecularRepresentationChoice,
            surrogateModelChoice=surrogateModelChoice,
            surrogateModelCheckPoints=surrogateModelCheckPoints,
            trainedSurrogateModelCheckPoints=trainedSurrogateModelCheckPoints,
            acquisitionFunctionChoice=acquisitionFunctionChoice,
            virtualLibraryLoops=virtualLibraryLoops,
            acquisitionBatchSize=virtualLibraryAcquisitionBatchSize,
            currentEpoch=currentEpoch,
            direction=direction,
            pathToAcquiredCompounds=pathToOracleAcquiredCompoundsDir,
            pathToNonAcquiredCompounds=pathToSurrogateAcquiredCompoundsDir,
            trainingPoolLimit=trainingPoolLimit,
            trainingPoolMaxCompounds=trainingPoolMaxCompounds,
            trainingPoolEpochStart=trainingPoolEpochStart,
            relative_fraction=relative_fraction,
            num_cpus=num_cpus,
            n_lig=n_lig,
            subsample=subsample,
            mpo_acquisition=mpo_acquisition, 
            ucb_beta_value=ucb_beta_value,
        )

        stop = time.perf_counter()
        logging.info(
            f'Finished virtual screen active learning REINVENT in {np.round(stop-start, 2)}(s), Acquired scores: {oracle_acquired_data.agg({"scores": ["min", "max", "mean", "median"]})}'
        )

    if "oracle_acquired_data" in locals():
        oracle_acquired_data['weight'] = 1.0 

    if "surrogate_acquired_data" in locals():
        surrogate_acquired_data["scores"] = surrogate_acquired_data[
            "predictions"
        ].to_numpy()

        surrogate_acquired_data["weight"] = [predicted_weights] * len(
            surrogate_acquired_data
        )
        # assign weights based on model confidence
        surrogate_acquired_data = _scale_to_confidence(
            predicted_weights, confidence_cutoff, surrogate_acquired_data
        )

        _save_dataframe(
                dirPath=pathToOracleAcquiredCompoundsDir,
                dataframe = surrogate_acquired_data,
                currentEpoch=currentEpoch,
                )
            

        surrogate_acquired_data["scores"] = [
            x if y > 0 else 0
            for x, y in zip(
                surrogate_acquired_data["scores"].to_numpy(),
                surrogate_acquired_data["weight"].to_numpy(),
            )
        ]
        full_frame = pd.concat([oracle_acquired_data, surrogate_acquired_data])
    else:
        full_frame = oracle_acquired_data
        full_frame["weight"] = [1] * len(oracle_acquired_data)

    ######### SAVING ACQUIRED COMPOUNDS ##########
    _save_dataframe(
        dirPath=pathToOracleAcquiredCompoundsDir,
        dataframe = oracle_acquired_data,
        currentEpoch=currentEpoch,
        )
    

    # Increment the counter
    docking_counter += len(oracle_acquired_data)
    logging.info(f"Number of docking calls performed so far {docking_counter}")

    # Open the file again and write the new value of the counter
    with open(docking_counter_file_path, "w") as f:
        f.write(str(docking_counter))

    if (
        "normal_reinvent_non_acquired" in locals()
        and normal_reinvent_non_acquired is not None
    ):
        normal_reinvent_non_acquired["weight"] = [0] * len(normal_reinvent_non_acquired)
        normal_reinvent_non_acquired["scores"] = [0] * len(normal_reinvent_non_acquired)
        full_frame = pd.concat([full_frame, normal_reinvent_non_acquired])

    if "failed_smiles" in locals() and failed_smiles is not None:
        failed_smiles["scores"] = 0
        failed_smiles["weight"] = 1
        full_frame = pd.concat([full_frame, failed_smiles])
        logging.debug(
            f"These smileWithMol failed\n {failed_smiles['original_smiles'].to_numpy()})"
        )

    if "non_unique_smileWithMol" in locals() and non_unique_smileWithMol is not None:
        non_unique_smileWithMol["scores"] = 0
        non_unique_smileWithMol["weight"] = 0

        index_dic = {
            k: v
            for k, v in zip(
                oracle_acquired_data["original_smiles"].to_numpy(),
                oracle_acquired_data["scores"].to_numpy(),
            )
        }

        scores_for_duplicates = [
            index_dic.get(smile, 0.0)
            for smile in non_unique_smileWithMol["original_smiles"].to_numpy()
        ]

        non_unique_smileWithMol["scores"] = scores_for_duplicates
        non_unique_smileWithMol["weights"] = [
            1 if x > 0.0 else 0 for x in scores_for_duplicates
        ]

        full_frame = pd.concat([full_frame, non_unique_smileWithMol])

        logging.debug(
            f"Compounds that were not unique\n {non_unique_smileWithMol['original_smiles'].drop_duplicates().to_numpy()}"
        )
        logging.debug(f"Scores for duplicates are:\n {scores_for_duplicates}\n")


    full_frame = full_frame.sort_index()
    scores = full_frame["scores"].to_numpy()
    weights = full_frame["weight"].to_numpy()
    

    # If desired this function adds Gaussian Noise to Data
    if noise_level > 0.0 and noise_level is not False: 
        # oracle_acquired_data["noise_scores"] = _add_noise(
        #     noise_level, oracle_acquired_data["scores"].to_numpy()
        # )
        logging.info('Adding Noise To Data')
        scores = [_add_noise(noise_level, y) if x > 0.0 else _add_noise(0.0, y) for x, y in zip(weights, scores)]

    ##Validate that the ordering of smiles are still the same\f
    order_index = [
        smiles_order_dictionary.get(smile, False)
        for smile in full_frame["original_smiles"].to_numpy()
    ]
    logging.debug(f"The order of the original smiles is below\n{order_index}\n")
    logging.debug(
        f"Returning {len([x for x in scores if x != 0])} non zero scores and {len([x for x in weights if x != 0])} non zero weights"
    )
    print(weights)
    return scores, weights


def _scale_to_confidence(predicted_weights, confidence_cutoff, surrogate_acquired_data):
    if np.float32(confidence_cutoff) != 0.0:
        surrogate_acquired_data["weight"] = _make_the_weight_scaled_to_confidence(
            predictions=surrogate_acquired_data["predictions"].to_numpy(),
            confidence=surrogate_acquired_data["confidence"].to_numpy(),
            confidence_cutoff=confidence_cutoff,
            return_value=predicted_weights,
        )

        surrogate_acquired_data["scores"] = [0] * len(surrogate_acquired_data)

        predictions = [
            x if y else 0
            for x, y in zip(
                surrogate_acquired_data["predictions"].to_numpy(),
                surrogate_acquired_data["weight"].to_numpy(),
            )
        ]

        surrogate_acquired_data["scores"] = predictions

        logging.debug(
            f'This is the weights based on cutoff {confidence_cutoff}, {surrogate_acquired_data["weight"].to_numpy()}'
        )
        logging.debug(
            f'This is the cumulative weight of predicted smileWithMol {surrogate_acquired_data["weight"].to_numpy().sum()}'
        )

    return surrogate_acquired_data


def normal_reinvent(
    smiles_dataframe: str,
    oracleObject: str,
):
    originalSmiles = smiles_dataframe["original_smiles"].to_numpy()

    smiles_dataframe["scores"] = _query_oracle(
        originalSmiles=originalSmiles, oracleObject=oracleObject
    )

    return smiles_dataframe


def reinvent_with_active_learning(
    currentEpoch: int,
    smileWithMol: str,
    trainingPoolEpochStart: int,
    trainingPoolLimit: int,
    trainingPoolMaxCompounds: int,
    pathToOracleAcquiredCompoundsDir: str,
    pathToSurrogateAcquiredCompoundsDir: str,
    surrogateModelCheckPoints: str,
    trainedSurrogateModelCheckPoints: str,
    molecularRepresentationObject: str,
    oracleObject: str,
    surrogateModelObject: str,
    acquisitionFunctionObject: str,
    acquisitionBatchSize: int,
    direction: str,
    mpo_acquisition: bool, 
    ucb_beta_value: np.float32
):
    trainingPoolCompounds = _read_in_training_pool(
        pathToAcquiredCompounds=pathToOracleAcquiredCompoundsDir,
        currentEpoch=currentEpoch,
        trainingPoolEpochStart=trainingPoolEpochStart,
        trainingPoolLimit=trainingPoolLimit,
        trainingPoolMaxCompounds=trainingPoolMaxCompounds,
    )

    scoresTrain = trainingPoolCompounds["scores"].to_numpy()
    smiles_train = trainingPoolCompounds["original_smiles"].to_numpy()

    molecularRepresentationsTrain = _generate_molecularRepresentations(
        molecularRepresentationObject=molecularRepresentationObject,
        smiles=smiles_train,
    )

    trainedSurrogateModel, modelSavePath, trainedModelSavePath = _train_surrogate_model(
        surrogateModelObject=surrogateModelObject,
        surrogateModelCheckPoints=surrogateModelCheckPoints,
        trainedSurrogateModelCheckPoints=trainedSurrogateModelCheckPoints,
        currentEpoch=currentEpoch,
        molecularRepresentations=molecularRepresentationsTrain,
        scores=scoresTrain,
    )

    smiles_test = smileWithMol["original_smiles"].to_numpy()

    molecularRepresentations = _generate_molecularRepresentations(
        molecularRepresentationObject=molecularRepresentationObject,
        smiles=smiles_test,
    )

    # TODO The trained surrogate model object is currently read in from a saved directory, this is not the most efficient, ultimately it would be better to keep this as a continous process rather than requiring state readouts
    (
        smileWithMol["predictions"],
        smileWithMol["confidence"],
    ) = _predict_with_surrogate_model(
        molecularRepresentations=molecularRepresentations,
        modelSavePath=modelSavePath,
        trainedModelSavePath=trainedModelSavePath,
    )

    toAcquireDataframe, dontAcquireDataframe = _partitionCompounds(
        acquisitionFunctionObject=acquisitionFunctionObject,
        acquisitionBatchSize=acquisitionBatchSize,
        smileWithMolWithPredictionsDataFrame=smileWithMol,
        direction=direction,
        mpo_acquisition=mpo_acquisition,
        ucb_beta_value=ucb_beta_value
    )

    toAcquireDataframe["scores"] = _query_oracle(
        originalSmiles=toAcquireDataframe["original_smiles"].to_numpy(),
        oracleObject=oracleObject,
    )

    return toAcquireDataframe, dontAcquireDataframe


def REINVENT_virtual_library_screen(
    pathToOutDir: str,
    smileWithMol: str,
    oracleChoice: str,
    molecularRepresentationChoice: object,
    surrogateModelChoice: object,
    acquisitionFunctionChoice: object,
    surrogateModelCheckPoints: os.path,
    trainedSurrogateModelCheckPoints: os.path,
    pathToAcquiredCompounds: os.path,
    pathToNonAcquiredCompounds: os.path,
    trainingPoolEpochStart: int,
    trainingPoolLimit: int,
    trainingPoolMaxCompounds: int,
    confPath: os.path,
    propertyName: str,
    relative_fraction: np.float32,
    n_lig: np.float32,
    num_cpus: int,
    ucb_beta_value: np.float32,  
    virtualLibraryLoops: int = 10,
    acquisitionBatchSize: int = 5,
    currentEpoch=0,
    direction="positive",
    subsample=False,
    mpo_acquisition = False,
):
    molecularRepresentationClass = molecular_representations_class()
    molecularRepresentationObject = molecularRepresentationClass.choice(
        molecularRepresentationChoice=molecularRepresentationChoice
    )

    surrogateModelClass = surrogate_models_class()
    surrogateModelObject = surrogateModelClass.choice(
        surrogateModelChoice=surrogateModelChoice,
        molecularRepresentationChoice=molecularRepresentationChoice,
    )

    acquisitionFunctionClass = acquisition_functions_class(
        relative_fraction=relative_fraction,
        num_cpus=num_cpus,
        n_lig=n_lig,
    )
    acquisitionFunctionObject = acquisitionFunctionClass.choice(
        acquisitionFunctionChoice=acquisitionFunctionChoice,
    )

    oracleClass = oracles_class(
        confPath=confPath,
        propertyName=propertyName,
        currentEpoch=currentEpoch,
        direction=direction,
    )
    
    oracleObject = oracleClass.choice(oracleChoice=oracleChoice)
    smileWithMol = smileWithMol.sample(frac=1)
    smiles_test = smileWithMol["original_smiles"].to_numpy()

    smileWithMol["molecular_representations"] = _generate_molecularRepresentations(
        molecularRepresentationObject=molecularRepresentationObject,
        smiles=smiles_test,
        )

    trainingPoolSmiles = [] 
    trainingPoolMolecularRepresentations = [] 
    trainingPoolScores = [] 
    CollatedFrameList = []
    
    previous_epoch_compounds = os.path.join(pathToAcquiredCompounds, f"{currentEpoch-1}_Compounds.parquet") 
    if os.path.exists(previous_epoch_compounds):   
        if subsample:         
            trainingPoolCompounds = _subsample_training_pool(
                pathToAcquiredCompounds=pathToAcquiredCompounds,
                subsample_size=trainingPoolMaxCompounds,
                ucb_beta_value=ucb_beta_value,
        )
        else:
            trainingPoolCompounds = _read_in_training_pool(
                pathToAcquiredCompounds=pathToAcquiredCompounds,
                currentEpoch=currentEpoch,
                trainingPoolEpochStart=trainingPoolEpochStart,
                trainingPoolLimit=trainingPoolLimit,
                trainingPoolMaxCompounds=trainingPoolMaxCompounds,
            )
    
        trainingPoolSmiles.extend(trainingPoolCompounds['original_smiles'].values)
        trainingPoolMolecularRepresentations.extend(trainingPoolCompounds['molecular_representations'].values)
        trainingPoolScores.extend(trainingPoolCompounds['scores'].values)

    for loopEpoch in range(virtualLibraryLoops):

        logging.info(f"Starting loop {loopEpoch + 1} of {virtualLibraryLoops}")
        
        start = time.perf_counter()

        #If there is no training pool randomly partition compounds for acquisition.
        if len(trainingPoolSmiles) == 0: 
           
            logging.info(f"Performing Random Acquisiton of {acquisitionBatchSize} smileWithMol")

            toAcquireDataframe, dontAcquireDataframe = _random_partition(
                smileWithMol,
                acquisitionBatchSize=acquisitionBatchSize,
                relative_fraction=relative_fraction,
                num_cpus=num_cpus,
                ucb_beta_value=ucb_beta_value,
            )

            toAcquireDataframe["scores"] = _query_oracle(
                originalSmiles=toAcquireDataframe["original_smiles"].to_numpy(), oracleObject=oracleObject, loopEpoch=loopEpoch
            )
            
            trainedSurrogateModel, modelSavePath, trainedModelSavePath = _train_surrogate_model(
                surrogateModelObject=surrogateModelObject,
                surrogateModelCheckPoints=surrogateModelCheckPoints,
                trainedSurrogateModelCheckPoints=trainedSurrogateModelCheckPoints,
                molecularRepresentations=toAcquireDataframe["molecular_representations"].to_list(),
                loopEpoch=(loopEpoch + 1),
                currentEpoch=currentEpoch,
                scores=toAcquireDataframe["scores"].to_list(),
            )            

            dontAcquireDataframe["predictions"], dontAcquireDataframe["confidence"] = _predict_with_surrogate_model(
                molecularRepresentations=dontAcquireDataframe["molecular_representations"].to_list(),
                modelSavePath=modelSavePath,
                trainedModelSavePath=trainedModelSavePath,
            )
            
            toAcquireDataframe["predictions"], toAcquireDataframe["confidence"] = _predict_with_surrogate_model(
                molecularRepresentations=toAcquireDataframe["molecular_representations"].to_list(),
                modelSavePath=modelSavePath,
                trainedModelSavePath=trainedModelSavePath,
            )

        if len(trainingPoolSmiles) != 0: 

            trainedSurrogateModel, modelSavePath, trainedModelSavePath = _train_surrogate_model(
                surrogateModelObject=surrogateModelObject,
                surrogateModelCheckPoints=surrogateModelCheckPoints,
                trainedSurrogateModelCheckPoints=trainedSurrogateModelCheckPoints,
                molecularRepresentations=trainingPoolMolecularRepresentations,
                loopEpoch=(loopEpoch + 1),
                currentEpoch=currentEpoch,
                scores=trainingPoolScores,
            )

            if 'dontAcquireDataframe' not in locals():
                dontAcquireDataframe = smileWithMol

            dontAcquireDataframe["predictions"], dontAcquireDataframe["confidence"] = _predict_with_surrogate_model(
                molecularRepresentations=dontAcquireDataframe["molecular_representations"].to_list(),
                modelSavePath=modelSavePath,
                trainedModelSavePath=trainedModelSavePath,
            )

            toAcquireDataframe, dontAcquireDataframe = _partitionCompounds(
                acquisitionFunctionObject=acquisitionFunctionObject,
                acquisitionBatchSize=acquisitionBatchSize,
                smileWithMolWithPredictionsDataFrame=dontAcquireDataframe,
                direction=direction,
                mpo_acquisition=mpo_acquisition,
                ucb_beta_value=ucb_beta_value
            )
        
            toAcquireDataframe["scores"] = _query_oracle(
                originalSmiles=toAcquireDataframe["original_smiles"].to_numpy(), oracleObject=oracleObject, loopEpoch=loopEpoch
                )


        end = time.perf_counter()

        logging.info(f"Completed a virtual library loop in {round(end-start, 2)}(s)")

        trainingPoolSmiles.extend(toAcquireDataframe["original_smiles"].values)
        trainingPoolMolecularRepresentations.extend(toAcquireDataframe["molecular_representations"].values)
        trainingPoolScores.extend(toAcquireDataframe["scores"].values)

        CollatedFrameList.append(toAcquireDataframe)

    if virtualLibraryLoops > 1:         
        CollatedFrame = pd.concat(CollatedFrameList)
    else: 
        CollatedFrame = CollatedFrameList[0]
    
    return CollatedFrame, dontAcquireDataframe


############## Helper Functions ############


def _make_the_data_bit_noisy(noise_level: np.float32, data_to_noiseify: list):
    mu, sigma = 0, noise_level
    noise = np.random.normal(mu, sigma, data_to_noiseify.shape)
    noisiest_data = data_to_noiseify + noise
    return noisiest_data


def _make_the_weight_scaled_to_confidence(
    predictions, confidence, confidence_cutoff, return_value
):
    # def is_it_one_or_zero(cutoff, value, return_value):
    #     if value <= cutoff:
    #         return return_value
    #     else:
    #         return 0

    # weights = [
    #     is_it_one_or_zero(confidence_cutoff, y / x, return_value)
    #     for x, y in zip(predictions, confidence)
    # ]

    weights = [return_value if conf < confidence_cutoff else 0 for conf in confidence]

    return weights


def _create_directories(pathToOutDir):
    compoundSavePath = os.path.join(pathToOutDir, "activeLearningCompounds")

    oracleAcquiredCompoundsDir = os.path.join(
        compoundSavePath, "oracleAcquiredCompounds"
    )

    surrogateAcquiredCompoundsDir = os.path.join(
        compoundSavePath, "surrogateAcquiredCompounds"
    )

    modelSavePath = os.path.join(pathToOutDir, "surrogateCheckpoints")
    surrogateModelCheckPoints = os.path.join(modelSavePath, "surrogateModelCheckPoints")

    trainedSurrogateModelCheckPoints = os.path.join(
        modelSavePath, "trainedSurrogateModelCheckPoints"
    )

    reinventBatch = os.path.join(pathToOutDir, "reinventBatch")
    reinventBatchFile = os.path.join(reinventBatch, "reinventCompounds.csv")

    for dir in [
        compoundSavePath,
        modelSavePath,
        pathToOutDir,
        oracleAcquiredCompoundsDir,
        surrogateAcquiredCompoundsDir,
        surrogateModelCheckPoints,
        trainedSurrogateModelCheckPoints,
        reinventBatch,
    ]:
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass

    return (
        reinventBatchFile,
        oracleAcquiredCompoundsDir,
        surrogateAcquiredCompoundsDir,
        surrogateModelCheckPoints,
        trainedSurrogateModelCheckPoints,
    )


def _create_virtual_library_directory(pathToOutDir):
    virtualLibraryDir = os.path.join(pathToOutDir, "virtualLibrary")
    inputDirectory = os.path.join(virtualLibraryDir, "virtualLibraryInput")
    outputDirectory = os.path.join(virtualLibraryDir, "virtualLibraryOutput")

    virtualLibrarytrainingPoolDir = os.path.join(
        outputDirectory, "virtualLibraryTrainingPool"
    )
    virtualLibrarytoAcquireDir = os.path.join(
        outputDirectory, "virtualLibrarytoAcquireDir"
    )
    virtualLibrarynotAcquiredDir = os.path.join(
        outputDirectory, "virtualLibrarynotAcquiredDir"
    )
    virtualLibraryAcquiredDir = os.path.join(
        outputDirectory, "virtualLibraryAcquiredDir"
    )

    for dir in [
        virtualLibraryDir,
        pathToOutDir,
        inputDirectory,
        outputDirectory,
        virtualLibrarytrainingPoolDir,
        virtualLibrarynotAcquiredDir,
    ]:
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass

    return (
        inputDirectory,
        virtualLibrarytrainingPoolDir,
        virtualLibrarynotAcquiredDir,
    )


def _validate_smiles(originalSmiles):
    def catch(value):
        if value is None:
            return "Failed"
        else:
            return value

    rdKitMolObjects = [catch(Chem.MolFromSmiles(smile)) for smile in originalSmiles]
    smileWithMol = pd.DataFrame({"original_smiles": originalSmiles})
    smileWithMol["rdkit_mol_objects"] = rdKitMolObjects

    if "Failed" in smileWithMol["rdkit_mol_objects"]:
        index = smileWithMol["rdkit_mol_objects"] == "Failed"
        failed_smiles = smileWithMol.iloc[index]
        smileWithMol = smileWithMol[-index]
    else:
        failed_smiles = None

    return smileWithMol, failed_smiles


def _generate_molecularRepresentations(molecularRepresentationObject, smiles):
    start = time.perf_counter()

    logging.info(
        f"Starting generation of molecular representation for {len(smiles)} rdKitMolObjects"
    )
    molecularRepresentations = molecularRepresentationObject.calculateRepresentation(
        smiles
    )
    stop = time.perf_counter()
    logging.info(
        f"Generated {len(molecularRepresentations)} molecular representations in {np.round(stop-start, 2)}(s)"
    )
   
    return molecularRepresentations


def _query_oracle(originalSmiles, oracleObject, loopEpoch=False):
    start = time.perf_counter()
    scores = oracleObject.calculateScores(originalSmiles, loopEpoch)
    stop = time.perf_counter()

    logging.info(
        f"Generated {len(scores)} labels with a mean score: {np.round(np.mean(scores),2)} for {len(originalSmiles)} in {round(stop-start, 2)}(s)"
    )

    return scores


def _save_dataframe(dirPath, dataframe, currentEpoch):

    file_save_path = os.path.join(dirPath, f"{currentEpoch}_Compounds.parquet")
    
    columns_to_keep = ["original_smiles", "scores", "molecular_representations", "predictions", "confidence"]
    clean_dataframe = dataframe.drop([x for x in dataframe.columns if x not in columns_to_keep], axis=1, errors='ignore')

    clean_dataframe.to_parquet(file_save_path)
    
    return


def _read_in_training_pool(
    pathToAcquiredCompounds,
    currentEpoch,
    trainingPoolLimit,
    trainingPoolEpochStart,
    trainingPoolMaxCompounds,
):
    sortedDirectoryFiles = _sort_directory(pathToDir=pathToAcquiredCompounds)

    trainingPoolFiles = sortedDirectoryFiles

    if trainingPoolEpochStart:
        trainingPoolFiles = sortedDirectoryFiles[trainingPoolEpochStart:]

    if trainingPoolLimit:
        index = min(trainingPoolLimit, len(sortedDirectoryFiles))
        trainingPoolFiles = sortedDirectoryFiles[-index:]

    iterable_of_training_pool_files = [
        pd.read_parquet(os.path.join(pathToAcquiredCompounds, epochFileName))
        for epochFileName in trainingPoolFiles
    ]

    trainingPool = pd.concat(iterable_of_training_pool_files)

    if trainingPoolMaxCompounds:
        index = min(len(trainingPool), trainingPoolMaxCompounds)
        trainingPool = trainingPool[-index:]

    return trainingPool


def _subsample_training_pool(
    pathToAcquiredCompounds,
    subsample_size,
    ucb_beta_value
):
    start = time.perf_counter()
    
    sorted_directory = _sort_directory(pathToDir=pathToAcquiredCompounds)
    df_iter = [
        pd.read_parquet(os.path.join(pathToAcquiredCompounds, file_))
        for file_ in sorted_directory
    ]
    training_pool = pd.concat(df_iter)

    acquisitionBatchSize = np.floor(0.1 * subsample_size)
    acquisitionBatchSize = int(np.floor(min(acquisitionBatchSize, (0.1 * len(training_pool)))))

    training_data_ls = []

    subsample_epochs = 10
    for subsample_epoch in range(subsample_epochs):
        af_choice = "uncertain"
        if "surrogate_query_data" not in locals():
            surrogate_query_data = training_pool
            af_choice = "random"
        
        if acquisitionBatchSize == 0:
            break
        af = acquisition_functions_class(n_lig=0.0, relative_fraction=0.0, num_cpus=0.0)
        af = af.choice(af_choice)
        oracle_query_data, surrogate_query_data = af.partition(
            compoundDataFrame=surrogate_query_data,
            acquisitionBatchSize=acquisitionBatchSize,
            direction="positive",
            mpo_acquisition=False,
            ucb_beta_value = ucb_beta_value
        )
              
        training_data = oracle_query_data
        training_data_ls.append(oracle_query_data)
        if len(training_data_ls) > 1:
            training_data = pd.concat(training_data_ls)

        sm = surrogate_models_class()
        sm = sm.choice("randomforest", molecularRepresentationChoice="physchem")
       
        trained_model = sm.train(molecularRepresentations=[x for x in training_data["molecular_representations"].to_numpy()],
            scores=training_data["scores"].to_numpy(),
        )
       
        surrogate_query_data["predictions"], surrogate_query_data["confidence"] = sm.predict(molecularRepresentations=[x for x in surrogate_query_data["molecular_representations"].to_numpy()])

        rmse = np.mean(np.sqrt((surrogate_query_data["predictions"] - surrogate_query_data["scores"])** 2))

        logging.info(f"RMSE for subsampling round {subsample_epoch+1} is {rmse}")
        logging.debug(
            f"Diagnostics,\nTraining Pool Length: {len(training_data)},\nRemaining Data: {len(surrogate_query_data)}"
        )
    
    stop = time.perf_counter()
    logging.info(f'Finished active learning sub_sampling in {np.round(stop-start, 2)}(s)')

    return pd.concat(training_data_ls)


def _sort_directory(pathToDir):
    unsortedDirectoryFiles = os.listdir(pathToDir)
    sortedDirectoryFiles = sorted(
        unsortedDirectoryFiles, key=lambda x: float(x.split("_")[0])
    )
    try:
        sortedDirectoryFiles = sorted(
            sortedDirectoryFiles, key=lambda x: int(x.split("_")[1])
        )
    except:
        pass

    return sortedDirectoryFiles


def _train_surrogate_model(
    surrogateModelObject,
    surrogateModelCheckPoints,
    trainedSurrogateModelCheckPoints,
    currentEpoch,
    molecularRepresentations,
    scores,
    loopEpoch=False,
    removeTempCheckpoints=True,
):
    start = time.perf_counter()

    trainedSurrogateModel = surrogateModelObject.train(
        molecularRepresentations=molecularRepresentations, scores=scores
    )

    if loopEpoch:
        modelSavePath = os.path.join(
            surrogateModelCheckPoints,
            f"{currentEpoch}_{loopEpoch}_surrogateModelObject.sav",
        )
        trainedModelSavePath = os.path.join(
            trainedSurrogateModelCheckPoints,
            f"{currentEpoch}_{loopEpoch}_trainedSurrogateModel.sav",
        )
    else:
        modelSavePath = os.path.join(
            surrogateModelCheckPoints, f"{currentEpoch}_surrogateModelObject.sav"
        )
        trainedModelSavePath = os.path.join(
            trainedSurrogateModelCheckPoints,
            f"{currentEpoch}_trainedSurrogateModel.sav",
        )

    if removeTempCheckpoints:
        [
            os.remove(os.path.join(surrogateModelCheckPoints, pathA))
            for pathA in os.listdir(surrogateModelCheckPoints)
        ]
        [
            os.remove(os.path.join(trainedSurrogateModelCheckPoints, pathB))
            for pathB in os.listdir(trainedSurrogateModelCheckPoints)
        ]

    with open(modelSavePath, "wb") as f:
        pickle.dump(surrogateModelObject, f, pickle.HIGHEST_PROTOCOL)
    with open(trainedModelSavePath, "wb") as f:
        pickle.dump(trainedSurrogateModel, f, pickle.HIGHEST_PROTOCOL)

    stop = time.perf_counter()

    logging.info(
        f"Completed Training Surrogate Model with {len(molecularRepresentations)} smileWithMol in {round(stop-start, 2)}(s)"
    )

    return trainedSurrogateModel, modelSavePath, trainedModelSavePath


def _predict_with_surrogate_model(
    # surrogateModelObject, #trainedSurrogateModel,
    modelSavePath,
    trainedModelSavePath,
    molecularRepresentations,
    removeTempCheckpoints=True,
):
    with open(modelSavePath, "rb") as f:
        surrogateModelObject = pickle.load(f)
    with open(trainedModelSavePath, "rb") as f:
        trainedSurrogateModel = pickle.load(f)

    predictions, confidence = surrogateModelObject.predict(
        molecularRepresentations=molecularRepresentations,
        trained_surrogate=trainedSurrogateModel,
    )

    logging.info(
        f"Mean prediction: {np.round(np.mean(predictions),3)}, Mean confidence: {np.round(np.mean(confidence),3)}, Number Compounds: {len(molecularRepresentations)}"
    )

    return predictions, confidence


def _partitionCompounds(
    acquisitionFunctionObject,
    acquisitionBatchSize,
    smileWithMolWithPredictionsDataFrame,
    direction,
    mpo_acquisition,
    ucb_beta_value
):
    toAcquireDataframe, dontAcquireDataframe = acquisitionFunctionObject.partition(
        acquisitionBatchSize=acquisitionBatchSize,
        compoundDataFrame=smileWithMolWithPredictionsDataFrame,
        direction=direction,
        mpo_acquisition=mpo_acquisition,
        ucb_beta_value = ucb_beta_value
    )

    toAcquireDataframe["weight"] = 1
    dontAcquireDataframe["weight"] = 0

    return toAcquireDataframe.sort_index(), dontAcquireDataframe.sort_index()


def _random_partition(
    compoundDataFrame, acquisitionBatchSize, relative_fraction, num_cpus, ucb_beta_value
):
    if acquisitionBatchSize == len(compoundDataFrame):
        return compoundDataFrame, None

    randomAcquisitionFunctionClass = acquisition_functions_class(
        relative_fraction=0, n_lig=0.0, num_cpus=False
    )
    randomAcquisitionFunctionObject = randomAcquisitionFunctionClass.choice(
        acquisitionFunctionChoice="random",
    )

    (
        toAcquireDataframe,
        dontAcquireDataframe,
    ) = randomAcquisitionFunctionObject.partition(
        compoundDataFrame=compoundDataFrame,
        acquisitionBatchSize=acquisitionBatchSize,
        direction=False,
        mpo_acquisition=False,
        ucb_beta_value = ucb_beta_value
    )
    return toAcquireDataframe, dontAcquireDataframe


def _virtual_library_screen_loop(
    oracleObject,
    surrogateModelObject,
    acquisitionFunctionObject,
    trainingPoolSmiles,
    trainingPoolMolecularRepresentations,
    trainingPoolScores,
    dontAcquireDataframe,
    surrogateModelCheckPoints,
    trainedSurrogateModelCheckPoints,
    modelSavePath,
    trainedModelSavePath,
    currentEpoch,
    loopEpoch,
    direction,
    acquisitionBatchSize,
    mpo_acquisition,
    ucb_beta_value
):
    start = time.perf_counter()

    trainedSurrogateModel, modelSavePath, trainedModelSavePath = _train_surrogate_model(
        surrogateModelObject=surrogateModelObject,
        surrogateModelCheckPoints=surrogateModelCheckPoints,
        trainedSurrogateModelCheckPoints=trainedSurrogateModelCheckPoints,
        molecularRepresentations=trainingPoolMolecularRepresentations,
        loopEpoch=(loopEpoch + 1),
        currentEpoch=currentEpoch,
        scores=trainingPoolScores,
    )

    molecularRepresentationsNotAcquired = dontAcquireDataframe["molecular_representations"].to_numpy()
    
    predictions, confidence = _predict_with_surrogate_model(
        molecularRepresentations=molecularRepresentationsNotAcquired,
        modelSavePath=modelSavePath,
        trainedModelSavePath=trainedModelSavePath,
    )

    dontAcquireDataframe["predictions"] = predictions
    dontAcquireDataframe["confidence"] = confidence

    smilesNotAcquired = dontAcquireDataframe["original_smiles"].to_numpy()
    molecularRepresentationsNotAcquired = dontAcquireDataframe[
        "molecular_representations"
    ].to_numpy()

    toAcquireDataframe, dontAcquireDataframe = _partitionCompounds(
        acquisitionFunctionObject=acquisitionFunctionObject,
        acquisitionBatchSize=acquisitionBatchSize,
        smileWithMolWithPredictionsDataFrame=dontAcquireDataframe,
        direction=direction,
        mpo_acquisition=mpo_acquisition,
        ucb_beta_value=ucb_beta_value
    )

    smilesToAcquire = toAcquireDataframe["original_smiles"].to_numpy()

    scoresAcquired = _query_oracle(
        originalSmiles=smilesToAcquire, oracleObject=oracleObject, loopEpoch=loopEpoch
    )

    toAcquireDataframe["scores"] = scoresAcquired

    end = time.perf_counter()

    logging.info(f"Completed a virtual library loop in {round(end-start, 2)}(s)")

    trainingPoolSmiles.extend(toAcquireDataframe["original_smiles"].values)
    trainingPoolMolecularRepresentations.extend(
        toAcquireDataframe["molecular_representations"].values
    )
    trainingPoolScores.extend(toAcquireDataframe["scores"].values)

    return (
        trainingPoolSmiles,
        trainingPoolMolecularRepresentations,
        trainingPoolScores,
        toAcquireDataframe,
        dontAcquireDataframe,
        modelSavePath,
        trainedModelSavePath,
    )


def _virtual_screen_condition(
    currentEpoch,
    warmup,
    condition,
):
    if int(condition) == 0:
        return False
    if currentEpoch > warmup:
        if (currentEpoch % condition) == 0:
            return True
    else:
        return False
##update file so unison doesn't murder

def _add_noise(noise_level, scores):
    if np.float32(noise_level) != 0.0:
        #logging.info(f" these are pre noisified scores {scores[0:5]}")
        scores = _make_the_data_bit_noisy(
            noise_level=noise_level, data_to_noiseify=np.array(scores)
        )
        #logging.info(f" these are post noisified scores {scores[0:5]}")
    return scores


def _initialise_arguments(
    currentEpoch,
    originalSmiles,
    molecularRepresentationChoice,
    oracleChoice,
    confPath,
    propertyName,
    surrogateModelChoice,
    acquisitionFunctionChoice,
    direction,
    relative_fraction,
    reinventBatchFile,
    n_lig,
    num_cpus,
):
    originalSmiles_dataframe = pd.DataFrame({"original_smiles": originalSmiles})
    originalSmiles_dataframe.to_csv(reinventBatchFile, index=False)

    molecularRepresentationClass = molecular_representations_class()
    molecularRepresentationObject = molecularRepresentationClass.choice(
        molecularRepresentationChoice=molecularRepresentationChoice
    )

    surrogateModelClass = surrogate_models_class()
    surrogateModelObject = surrogateModelClass.choice(
        surrogateModelChoice=surrogateModelChoice,
        molecularRepresentationChoice=molecularRepresentationChoice,
    )

    acquisitionFunctionClass = acquisition_functions_class(
        relative_fraction=relative_fraction, num_cpus=num_cpus, n_lig=n_lig
    )
    acquisitionFunctionObject = acquisitionFunctionClass.choice(
        acquisitionFunctionChoice=acquisitionFunctionChoice
    )

    oracleClass = oracles_class(
        confPath=confPath,
        propertyName=propertyName,
        direction=direction,
        currentEpoch=currentEpoch,
    )
    oracleObject = oracleClass.choice(oracleChoice=oracleChoice)
    return (
        originalSmiles_dataframe,
        molecularRepresentationObject,
        surrogateModelObject,
        acquisitionFunctionObject,
        oracleObject,
    )


def _drop_duplicates(smileWithMol: pd.DataFrame) -> pd.DataFrame:
    smileWithMolUnique = smileWithMol.drop_duplicates(subset="original_smiles")
    if len(smileWithMolUnique) == len(smileWithMol):
        smileWithMolNonUnique = None
        logging.debug(f"REINVENT produced {(smileWithMolNonUnique)} duplicated smiles")
    else:
        smileWithMolNonUnique = smileWithMol.drop(smileWithMolUnique.index)
        logging.info(
            f"REINVENT produced {len(smileWithMolNonUnique)} duplicated smiles"
        )

    return smileWithMolUnique, smileWithMolNonUnique

def _string_to_array(string):
    x = np.fromstring(string, dtype=np.float32)
    return x 
