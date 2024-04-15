from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentalVariablesEnum:
    PIP_URL = "PIP_URL"
    PIP_KEY = "PIP_KEY"
    PIP_GET_RESULTS = "PIP_GET_RESULTS"
    JAVA_HOME = "JAVA_HOME"
    CHEMAXON_HOME = "CHEMAXON_HOME"
    CHEMAXON_EXECUTOR_PATH = 'CHEMAXON_EXECUTOR_PATH'
