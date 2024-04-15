import os
import json
from pathlib import Path


CONFIG = {}

integration_config_path = Path(__file__).parent / 'integration_tests_config.json'
dummy_config_path = Path(__file__).parent / 'example.config.json'

path = integration_config_path if integration_config_path.exists() else dummy_config_path

with open(path) as f:
    json_input = f.read().replace('\r', '').replace('\n', '')
    CONFIG = json.loads(json_input)

# load environment variables from config - don't override if present 
for key, value in CONFIG.get('ENVIRONMENTAL_VARIABLES', {}).items():
    if key not in os.environ:
        os.environ[key] = value

ACTIVITY_REGRESSION = 'unittest_reinvent/fixtures/dummy_regression_model.pkl'

MAIN_TEST_PATH = CONFIG["MAIN_TEST_PATH"]

SMILES_SET_PATH = CONFIG.get('SMILES_SET_PATH')
SMILES_SET_LINK_INVENT_PATH = CONFIG.get('SMILES_SET_LINK_INVENT_PATH')
PRIOR_PATH = CONFIG.get('PRIOR_PATH')
MOLFORMER_PRIOR_PATH = CONFIG.get('MOLFORMER_PRIOR_PATH')
LIBINVENT_PRIOR_PATH = CONFIG.get('LIBINVENT_PRIOR_PATH')

LINK_INVENT_PRIOR_PATH = CONFIG.get("LINK_INVENT_PRIOR_PATH")
MOLFORMER_SMILES_SET_PATH = CONFIG.get("MOLFORMER_SMILES_SET_PATH")
