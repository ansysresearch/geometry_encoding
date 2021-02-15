from src.params import parse_arguments
from src.utils import TrainLogger, read_data, find_best_gpu, compute_perimeter_img, get_loss_func
from src.train import train
from src.test import test, test_exotic_shape
from src.generate_data import generate_dataset
from src.visualize import viz
