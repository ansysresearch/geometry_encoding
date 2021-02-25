import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='SDF 2d')

    # mode
    parser.add_argument('-m', '-mode', dest='mode', type=str, required=True,
                        help='choose either of "train", "test", "test exotic" or "visualize"')
    parser.add_argument('--model-flag', dest='model_flag', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                        help='see comments under src/utils/prepare_training_data function')
    # data parameters
    parser.add_argument('--n-obj', dest='n_obj', type=int, default=500,
                        help='initial batch of objects for training. multiplied 9x during augmentation')
    parser.add_argument('--img-res', dest='img_res', type=int, default=128,
                        help='image resolution')
    parser.add_argument('--data-folder', dest='data_folder', type=str, default='data/datasets/',
                        help='folder where data are stored')
    parser.add_argument('--dataset-id', dest='dataset_id', type=str, default='all',
                        help='name of dataset')
    parser.add_argument('--data-network-id', dest='data_network_id', type=str,
                        help='processor or compressor network when generating data for training compressor or evaluator')

    # network parameters
    parser.add_argument('--network-id', dest='net_id', type=str, default='UNet1',
                        help='id of the network. see network/network_lib.py')
    parser.add_argument('--data-type', dest='dtype', type=str, default='float32', choices=['float32, float64'],
                        help='data type can be either float32 or float 64')

    # training parameters
    parser.add_argument('--use-cpu', dest='use_cpu', type=int, default=0, choices=[0, 1],
                        help='if true, use cpu')
    parser.add_argument('--n-epochs', dest='n_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=25,
                        help='batch size')
    parser.add_argument('--save-every', dest='save_every', type=int, default=10,
                        help='save network every x epochs')
    parser.add_argument('--loss-fn', dest='loss_fn', type=str, default='l1',
                        help='loss function, choice of l1 or l2')
    parser.add_argument('--learning-rate', dest='lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--learning-rate-step', dest='lr_step', type=int,
                        help='learning rate scheduler step')
    parser.add_argument('--min-learning-rate', dest='lr_min', type=float, default=1e-6,
                        help='minimum learning rate')
    parser.add_argument('--plateau-patience', dest='lr_patience', type=int, default=3,
                        help='plateau patience parameter in learning rate scheduling')
    parser.add_argument('--plateau-factor', dest='lr_factor', type=float, default=0.2,
                        help='plateau factor parameter in learning rate scheduling')
    parser.add_argument('--validation-fraction', dest='val_frac', type=float, default=0.2,
                        help='portion of data that is used for validation')
    parser.add_argument('--checkpoints-directory', dest='ckpt_dir', type=str, default='outputs/',
                        help='folder that all data during train and test will be saved in.')
    parser.add_argument('--n-predict', dest='n_pred', type=int, default=10,
                        help='number of predictions during testing')
    parser.add_argument('--save-name', dest='save_name', type=str, default="",
                        help='tag added to runs files')
    parser.add_argument('--deeponet-npoints-per-pass', dest='deeponet_npoints_per_pass', type=int, default=10,
                        help='whatever')
    return parser.parse_args()


# data parameters
# NUMBER_OBJECTS = 500
# IMAGE_RESOLUTION = 128
# DATA_FOLDER      = 'data/datasets/'
# DATASET_ID       = 'all' + str(IMAGE_RESOLUTION)
#
# # training parameters
# NETWORK_ID                     = 'UNet2'  # see network/network_lib.py for more details.
# NUM_EPOCHS                     = 100
# SAVE_EVERY                     = 20
# BATCH_SIZE                     = 10
# DATA_TYPE                      = torch.float32
# LEARNING_RATE                  = 5e-4
# NETWORK_SAVE_NAME              = NETWORK_ID + '_' + DATASET_ID
# VALIDATION_FRACTION            = 0.2
# LEARNING_RATE_PLATEAU_PATIENCE = 3
# LEARNING_RATE_PLATEAU_FACTOR   = 0.2
# MINIMUM_LEARNING_RATE          = 1e-6
# CHECKPOINTS_DIRECTORY          = 'checkpoints/'
# NETWORK_SAVE_DIRECTORY         = CHECKPOINTS_DIRECTORY + 'networks/'
# RUNS_SAVE_DIRECTORY            = CHECKPOINTS_DIRECTORY + 'runs/'
#
# # prediction parameters
# COMPUTE_PREDICTIONS_NUM   = 10
# PREDICTION_SAVE_DIRECTORY = CHECKPOINTS_DIRECTORY + 'predictions/'
