from src.params import parse_arguments
from src.utils import (TrainLogger, read_data, read_data_deeponet, get_device, get_dtype, get_optimizer,
                       get_loss_func, get_save_name)
from src.train import train
from src.test import test
from src.generate_data import generate_dataset
from src.visualize import viz
