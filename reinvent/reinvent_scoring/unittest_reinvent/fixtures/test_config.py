import os
import json
from pathlib import Path
from funcy import get_in


CONFIG = {}

integration_config_path = Path(__file__).parent / 'integration_tests_config.json'
dummy_config_path = Path(__file__).parent / 'example.config.json'

path = integration_config_path if integration_config_path.exists() else dummy_config_path

with open(path) as f:
    json_input = f.read().replace('\r', '').replace('\n', '')
    # load the config with path and env variables in the absence of reinventcli's configuration
    CONFIG = json.loads(json_input)


# load environment variables from config - don't override if present 
for key, value in CONFIG.get('ENVIRONMENTAL_VARIABLES', {}).items():
    if key not in os.environ:
        os.environ[key] = value

# shortcuts to fixtures
ACTIVITY_REGRESSION = "unittest_reinvent/fixtures/dummy_regression_model.pkl"
ACTIVITY_CLASSIFICATION = "unittest_reinvent/fixtures/dummy_classification_model.pkl"

AZDOCK_UNITTEST_JSON = "unittest_reinvent/scoring_tests/fixtures/azdock_data/azdock_OpenEye.json"
AZDOCK_UNITTEST_OE_RECEPTOR_PATH = "unittest_reinvent/scoring_tests/fixtures/azdock_data/azdock_OpenEye_receptor.oeb"
DOCKSTREAM_UNITTEST_JSON = "unittest_reinvent/scoring_tests/fixtures/dockstream_data/dockstream_OpenEye.json"
DOCKSTREAM_UNITTEST_OE_RECEPTOR_PATH = "unittest_reinvent/scoring_tests/fixtures/dockstream_data/dockstream_OpenEye_receptor.oeb"
ICOLOS_UNITTEST_JSON = "unittest_reinvent/scoring_tests/fixtures/icolos_data/icolos_NIBR.json"
ICOLOS_UNITTEST_GRID_PATH = "unittest_reinvent/scoring_tests/fixtures/icolos_data/icolos_cox2_grid.zip"
ICOLOS_UNITTEST_NIBR_NEGATIVE_IMAGE = "unittest_reinvent/scoring_tests/fixtures/icolos_data/icolos_NIBR_negative_image.mol2"

MAIN_TEST_PATH = get_in(CONFIG, ["MAIN_TEST_PATH"])
SAS_MODEL_PATH = get_in(CONFIG, ["COMPONENT_SPECIFIC", "SAS_MODEL_PATH"])

# component specific
AZDOCK_ENV_PATH = get_in(CONFIG, ["COMPONENT_SPECIFIC", "AZDOCK", "AZDOCK_ENV_PATH"])
AZDOCK_DOCKER_SCRIPT_PATH = get_in(CONFIG, ["COMPONENT_SPECIFIC", "AZDOCK", "AZDOCK_DOCKER_SCRIPT_PATH"])
AZDOCK_DEBUG = get_in(CONFIG, ["COMPONENT_SPECIFIC", "AZDOCK", "AZDOCK_DEBUG"])

DOCKSTREAM_ENV_PATH = get_in(CONFIG, ["COMPONENT_SPECIFIC", "DOCKSTREAM", "DOCKSTREAM_ENV_PATH"])
DOCKSTREAM_DOCKER_SCRIPT_PATH = get_in(CONFIG, ["COMPONENT_SPECIFIC", "DOCKSTREAM", "DOCKSTREAM_DOCKER_SCRIPT_PATH"])
DOCKSTREAM_DEBUG = get_in(CONFIG, ["COMPONENT_SPECIFIC", "DOCKSTREAM", "DOCKSTREAM_DEBUG"])

ICOLOS_EXECUTOR_PATH = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ICOLOS", "ICOLOS_EXECUTOR_PATH"])
ICOLOS_DEBUG = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ICOLOS", "ICOLOS_DEBUG"])
ICOLOS_UNITTEST_NIBR_VALUES_KEY = "shape_similarity"

ROCS_SIMILARITY_TEST_DATA = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ROCS", "ROCS_SIMILARITY_TEST_DATA"])
ROCS_MULTI_SIMILARITY_TEST_DATA = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ROCS", "ROCS_MULTI_SIMILARITY_TEST_DATA"])
ROCS_HIGH_ENERGY_QRY = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ROCS", "ROCS_HIGH_ENERGY_QRY"])
ROCS_SHAPE_QUERY = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ROCS", "ROCS_SHAPE_QUERY"])
ROCS_SHAPE_QUERY_2 = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ROCS", "ROCS_SHAPE_QUERY_2"])
ROCS_SHAPE_QUERY_3 = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ROCS", "ROCS_SHAPE_QUERY_3"])
ROCS_SHAPE_QUERY_CFF = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ROCS", "ROCS_SHAPE_QUERY_CFF"])
ROCS_SHAPE_QUERY_BATCH = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ROCS", "ROCS_SHAPE_QUERY_BATCH"])
ROCS_CUSTOM_CFF = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ROCS", "ROCS_CUSTOM_CFF"])
ROCS_NEG_VOL_PROTEIN = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ROCS", "ROCS_NEG_VOL_PROTEIN"])
ROCS_NEG_VOL_LIG = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ROCS", "ROCS_NEG_VOL_LIG"])
ROCS_NEG_VOL_SQ = get_in(CONFIG, ["COMPONENT_SPECIFIC", "ROCS", "ROCS_NEG_VOL_SQ"])

CHEMAXON_DEBUG = get_in(CONFIG, ["COMPONENT_SPECIFIC", "CHEMAXON", "CHEMAXON_DEBUG"])
CHEMAXON_INPUT_MARKUSH_STRUCTURE_PATH = get_in(CONFIG, ["COMPONENT_SPECIFIC", "CHEMAXON", "CHEMAXON_INPUT_MARKUSH_STRUCTURE_PATH"])

MMP_DEBUG = get_in(CONFIG, ["COMPONENT_SPECIFIC", "MMP", "MMP_DEBUG"])
