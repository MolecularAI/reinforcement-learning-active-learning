# Reinforcement Learning with Active Learning

## Project Description
--- 
This repository houses the necessary source code for executing Reinforcement Learning in tandem with Active Learning.

This project was created to alleviate the rising computational burdens associated with the detailed affinity calculations required for *in-silico* screening of small-molecule drug candidates.

With this in mind, we designed an active learning system that uses an iterative selection process to identify informative samples, thereby directing the evolution of a reinforcement learning agent. This approach ensures that the agent is trained using only the most pertinent and informative examples, thereby reducing redundancy and computational wastage.

By leveraging a smaller set of high-quality samples, we are able to expedite the convergence process. This has resulted in substantial efficiency gains with respect to CPU wall time per lead identified.

## Table of Contents
--- 
1. [[#Project Description]]
2. [[#Installation Instructions]]
3. [[#Project Usage]]
4. [[#Licence]]

## Installation Instructions
--- 
We recommend using the [Miniconda — conda documentation](https://docs.conda.io/en/latest/miniconda.html)to set up a virtual environment for installing dependencies. 

To clone this repository, you'll need [Git](https://git-scm.com) installed on your computer.

From your command line:
- Open your terminal/command line
- Change the current working directory to the location where you want the cloned directory to be made.

```bash
cd location/to/your/folder
git clone https://github.com/MolecularAI/reinforcement-learning-active-learning
cd reinforcement-learning-active-learning
```

### ReInvent
--- 
To install ReInvent we create an environment and install a single dependency. 

```Bash
cd reinvent/reinventcli
conda env create -f reinvent.yml
```

Once the reinvent.v3.2 environment is set up, we need to install the reinvent scoring package

```bash
cd reinvent/reinvent_scoring
conda activate reinvent.v3.2
pip install -e . 
```

For more information please refer to the main repository:
[Reinvent](https://github.com/MolecularAI/Reinvent)

### Icolos
--- 
There is an Icolos build included with this repository. Create the Icolos environment

```bash
cd icolos
conda env create -f environment_min.yml
```

For more information please refer to the main repository:
[Icolos](https://github.com/MolecularAI/Icolos)

## Project Usage
--- 
There have been two significant modifications to the ReInvent code base: the integration of an active learning system, and the introduction of a relevance weighting mechanism for Reinforcement Learning (RL) policy updates. These enhancements provide a new layer of sophistication to the ReInvent environment, augmenting its capacity for machine learning tasks. The configuration files included in this release contain references to the docking and ROCS query files that were utilized during the system testing phase. These can serve as practical examples or starting points for your own experiments.

### Reinvent Scoring
---
```reinvent_scoring.reinvent_scoring.scoring.score_components.active_learning```

The active learning score component is furnished with the necessary tools to approximate a given oracle. As of now, it supports ROCS and ADV implementations. 

- `retrospectiveReinvent.py`: 
  - This Python script forms the main execution code for the active learning algorithm. It initiates and controls the learning process.
  
- `acquisition_functions.py`:
  - This module houses various strategies for compound selection. It includes classes implementing diverse selection methods such as random, upper confidence bound, greedy, and uncertainty-based selection.

- `molecular_representations.py`:
  - This file includes methods for converting compounds into features suitable for training machine learning models. These features include physchem properties, ecfp, hash_ecfp, avalon, and macc's keys representations.
  
- `oracles.py`:
  - This script is responsible for interacting with Icolos and retrieving the values needed for model training. Currently, it supports ADV and ROCS oracles.

- `surrogate_models.py`:
  - This module manages the training and inference of various machine learning models. It supports a variety of models, including Random Forest, XGBoost, Support Vector Regression (SVR), Gaussian Processes, and K-Nearest Neighbours.

### Reinvent Model
---
```reinvent.reinventcli.running_modes.reinforcement_learning```

We present an enhancement for REINVENT that assigns a unique weight value to each SMILES score component. This process involves multiplying the individual weights of each SMILES to derive a final cumulative weight. This final weight is then applied to the specific contribution of an individual SMILES, resulting in a weighted loss update.

This allows fine-tuning of the individual contribution of a specific SMILES during the gradient update process. Through this, users can better manage and optimize their gradient adjustments.

## Example 
--- 
Example configuration files are provided in the configs directory and the prepared files for docking and chemical structure comparison is found in the data folder.  
## Licence
--- 
