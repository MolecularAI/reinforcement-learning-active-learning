"""AiZynthFinder command-line API.

For AiZynthFinder documentation, see:
- https://github.com/MolecularAI/aizynthfinder

This implementations calls external command.
The external command, in turn, can call AiZynthFinder command-line interface,
or run CLI in parallel on a cluster, or even call REST API for AiZynth.
External commands are left out of this repository.

Simplest external command, with AiZynthFinder installed,
would be the following bash script::

  #!/usr/bin/env bash

  cp /dev/stdin > input.smi && \
  aizynthcli \
     --smiles input.smi \
     --config config.yml \
     --output output.json.gz \
     >>aizynth-stdout-stderr.txt 2>&1 && \
  zcat output.json.gz

"""

import json
import logging
import os
import pathlib
import subprocess
import tempfile
import shlex
from typing import List, Dict, Tuple

import yaml

from reinvent_scoring.scoring.score_components.aizynth.parameters import AiZynthParams

logger = logging.getLogger(__name__)

HITINVENT_PROFILES = "HITINVENT_PROFILES"


def mergedicts(a: Dict, b: Dict, path=None):
    """Merges b into a, in place"""
    # https://stackoverflow.com/a/7205107
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                mergedicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                formatted_path = ".".join(path + [str(key)])
                raise Exception(
                    f"Conflict at {formatted_path}: {a[key]} vs {b[key]}."
                    f" Full dicts: {a}, {b}."
                )
        else:
            a[key] = b[key]
    return a


def prepare_config(params: AiZynthParams) -> Tuple[Dict, str]:
    """Prepare AiZynthFinder config and command.

    This function constructs AiZynthFinder config, in the following order:
    - Lowest priority is base config.
    - Next priority are values in profiles, they overwrite base config.
    - Highest priority are inline values in Reinvent scoring component.

    :param params: Reinvent params
    :return: dict with config
    """

    if HITINVENT_PROFILES in os.environ:
        with open(os.environ[HITINVENT_PROFILES]) as fp:
            uiprofiles = json.load(fp)
    else:
        uiprofiles = None

    if params.base_aizynthfinder_config:
        base_aizynthfinder_config = params.base_aizynthfinder_config
    elif uiprofiles and "base_aizynthfinder_config" in uiprofiles:
        base_aizynthfinder_config = uiprofiles.get("base_aizynthfinder_config")
    else:
        raise ValueError(
            f"Missing base_aizynthfinder_config for AiZynth scoring component."
            f" Add it to scoring component parameters, "
            f" or to Hitinvent UI profiles file "
            f" specified by environmental variable {HITINVENT_PROFILES}."
        )

    if params.custom_aizynth_command:
        cmd = params.custom_aizynth_command
    elif uiprofiles and "custom_aizynth_command" in uiprofiles:
        cmd = uiprofiles.get("custom_aizynth_command")
    else:
        raise ValueError(
            f"Missing custom_aizynth_command for AiZynth scoring component."
            f" Add it to scoring component parameters, "
            f" or to Hitinvent UI profiles file "
            f" specified by environmental variable {HITINVENT_PROFILES}."
        )

    with open(base_aizynthfinder_config) as fp:
        config = yaml.safe_load(fp)

    if (params.stock_profile or params.reactions_profile) and not uiprofiles:
        raise ValueError(
            f"Profile requested, but UI profile file not specified"
            f" by environmental variable {HITINVENT_PROFILES}."
        )

    if params.stock_profile:
        mergedicts(
            config,
            uiprofiles["stock_profiles"][params.stock_profile]["config"],
        )

    if params.reactions_profile:
        mergedicts(
            config,
            uiprofiles["reactions_profiles"][params.reactions_profile]["config"],
        )

    mergedicts(
        config,
        {
            "finder": {
                "properties": {
                    "max_transforms": params.number_of_steps,
                    "time_limit": params.time_limit_seconds,
                }
            }
        },
    )

    if params.stock:
        mergedicts(config, {"stock": params.stock})

    if params.scorer:
        mergedicts(config, {"scorer": params.scorer})

    return config, cmd


def run_aizynth(smiles: List[str], params: AiZynthParams, epoch: int) -> Dict:
    """Run custom command that executes AiZynth.

    This method assumes that custom command is as following:
    - takes SMILES strings, one per line, from stdin
    - writes output JSON to stdout
    - expects config.yml in the current directory
    Custom command can do anything else:
    - can create any additional files in the current directory
    - can call slurm, call REST API, wait, block etc

    This script creates a temporary directory inside current directory,
    and executes the command in that directory.
    """

    config, cmd = prepare_config(params)

    tmpdir = tempfile.mkdtemp(
        prefix=f"reinvent-aizynthfinder-{epoch}-", dir=os.getcwd()
    )

    configpath = pathlib.Path(tmpdir) / "config.yml"
    with open(configpath, "wt") as f:
        yaml.dump(config, f)

    input = "\n".join(smiles)
    result = subprocess.run(
        shlex.split(cmd),
        input=input,
        text=True,
        capture_output=True,
        shell=False,
        cwd=tmpdir,
    )
    if result.returncode != 0:
        logger.warning(
            f"AiZynth process returned non-zero returncode ({result.returncode})."
            f" Stderr:\n{result.stderr}"
        )
    out = json.loads(result.stdout)
    return out
