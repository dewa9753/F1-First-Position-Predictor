"""
lib.settings is a global settings module for the configuration of the main classifier.
"""
## PATH SETTINGS
DATA_NAME = 'jtrotman/formula-1-race-data'
ORIGINAL_DATA_ROOT = 'data_original'
DATA_ROOT = 'data'
MODEL_ROOT = 'model'

## DATA SETTINGS
MODEL_NAME = 'model_gb_clf'
RANDOM_STATE = 7
TEST_SIZE = 0.2
CV_FOLD = 5
N_JOBS = -1
