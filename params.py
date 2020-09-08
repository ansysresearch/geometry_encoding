import torch

# data parameters
IMAGE_RESOLUTION = 64
DATA_FOLDER      = "data/datasets/"
DATASET_ID       = "scipy_all" + str(IMAGE_RESOLUTION)

# training parameters
NETWORK_ID                     = "UNet64"
NUM_EPOCHS                     = 100
SAVE_EVERY                     = 20
BATCH_SIZE                     = 50
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
COMPUTE_PREDICTIONS_NUM   = 5
PREDICTION_SAVE_DIRECTORY = CHECKPOINTS_DIRECTORY + "predictions/"
