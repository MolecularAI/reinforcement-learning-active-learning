## AL Parameters

This file contains active learning parameters and their definitions, to help explain what changing each config will do.

  
- "al_output" 
	- Defines the output file path, where you would like all run outputs to be deposited
- "warmup"
	- Defines the number of rounds of RL before the AL model is used to approximate scores 
- "trainingPoolEpochStart"
	- From which epoch the training pool should begin to populate as its lower bound 
- "trainingPoolLimit"
	- Defines the last epoch at which the training pool will be populated from
- "trainingPoolMaxCompounds"
	- The maximum number of compounds to populate the training pool with, always uses latest compounds
- "molecularRepresentationChoice" 
	- "Physchem", "ECFP", the representation to use for compounds when training the ML model
- "oracleChoice"
	-  Defines the interface that will be used for communicating with the scoring function, can directly score compounds if it's setup in the scoring class
- "confPath"
	- Used when the oracle choice is "icolos", allows you to define the icolos configuration file 
- "propertyName"
	- The name of the returned property from icolos that contains the scores, expects it as an SDF tag
- "surrogateModelChoice":
	- Name of the model that should be used 
- "acquisitionFunctionChoice"
	- Name of the acquisition function that should be used
- "acquisitionBatchSize"
	- Number of compounds to acquire per epoch
- "direction" 
	- positive or negative, positive indicates that large scores are better
+ "loopCondition"
	+ How often, every N epochs, to perform multiple loops as opposed to a single loop
+ "virtualLibraryAcquisitionBatchSize"
	+ How many compounds to acquire per loop
+ "virtualLibraryLoops"
	+ How many loops to perform at each loop condition
+ "predicted_weights"
	+ A decimal value between 0 and 1 that indicates the relative importance of a predicted value for updating REINVENT's weights 
- "noise_level"
	- Not much practical use, adds noise to the final scores produced for each compound to simulate inaccuracy in the model 
- "confidence_cutoff"
	- Defines the confidence at which to apply the predicted_weights value, i.e. if the model is below a certain confidence threshold, apply the predicted_weights value else 0.
- "relative_fraction"
	- How close compounds must be in predicted score to the top compound for acquisition. The idea being to acquire only the top best compounds and ignore bad compounds.
- "drop_duplicate_smiles"
	- REINVENT sometimes produces duplicated compounds within the same epoch, this caches their scores and only computes the score for a singe compound if True. 
- "n_lig"
	- This defines the score of the native ligand, the idea being that any with a predicted score greater than the native ligand should be acquired.
- "num_cpus"
	- If you define a number of cpu's then the acquisition will use multiples of that definition for efficiency
- "max_docking_calls"
	- If you define a max number of docking calls, then the AL system will force the RL agent to terminate after a certain number of calls. This is not advised generally, but can be useful for quick testing.
- "sub_sample"
	- Construct the training pool using ML method [adaptive subsampling]([Improving Molecular Machine Learning Through Adaptive Subsampling with Active Learning | Theoretical and Computational Chemistry | ChemRxiv | Cambridge Open Engage](https://chemrxiv.org/engage/chemrxiv/article-details/63e5c76e1d2d18406337135d))
- "mpo_acquisition" 
	- This allows for acquisition based on the multi-parameter objective score rather than a single score - this function is currently hard coded for a given set of scoring components, and is therefore not suitable for general use, it is retained here for reproducibility.
- "ucb_beta_value"
	- When using upper confidence bound, define the scalar variable that weights the strength of the confidence values in determining which compounds to acquire. If set to 0 is equivalent to greedy acquisition.
- "transformation"
	- Values are transformed between 0-1 for RL update, define the criterion for this.