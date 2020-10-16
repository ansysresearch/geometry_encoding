import torch

# data parameters
NUMBER_OBJECTS = 500
IMAGE_RESOLUTION = 128
DATA_FOLDER      = "data/datasets/"
DATASET_ID       = "all" + str(IMAGE_RESOLUTION)

# training parameters
NETWORK_ID                     = "CNN1"  # see network/network_lib.py for more details.
NUM_EPOCHS                     = 100
SAVE_EVERY                     = 20
BATCH_SIZE                     = 10
DATA_TYPE                      = torch.float32
LEARNING_RATE                  = 5e-4
NETWORK_SAVE_NAME              = NETWORK_ID + "_" + DATASET_ID
VALIDATION_FRACTION            = 0.2
LEARNING_RATE_PLATEAU_PATIENCE = 3
LEARNING_RATE_PLATEAU_FACTOR   = 0.2
MINIMUM_LEARNING_RATE          = 1e-6
CHECKPOINTS_DIRECTORY          = "checkpoints/"
NETWORK_SAVE_DIRECTORY         = CHECKPOINTS_DIRECTORY + "networks/"
RUNS_SAVE_DIRECTORY            = CHECKPOINTS_DIRECTORY + "runs/"

# prediction parameters
COMPUTE_PREDICTIONS_NUM   = 10
PREDICTION_SAVE_DIRECTORY = CHECKPOINTS_DIRECTORY + "predictions/"
