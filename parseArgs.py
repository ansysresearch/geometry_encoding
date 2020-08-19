import time
import argparse

parser = argparse.ArgumentParser()
default_name = time.ctime().replace(" ", "_").replace(":", "")

parser.add_argument("-nw", "--network-id", type=int, help="network id", default=1)
parser.add_argument("-s", "--save-name", type=str, help="save name", default=default_name)
parser.add_argument("-d", "--dataset-id", type=str, help="dataset id", default="1")
parser.add_argument("-ne", "--num-epochs", type=int, help="number of epochs", default=100)
parser.add_argument("-se", "--save-every", type=int, help="save every", default=10)
parser.add_argument("-b", "--batch-size", type=int, help="batch size", default=250)
parser.add_argument("-lr", "--learning-rate", type=float, help="learning rate", default=0.01)
parser.add_argument("-chkdir", "--checkpoint-dir", type=str, help=" checkpoint directory", default="checkpoints/")
parser.add_argument("-imgres", "--img-resolution", type=int, help="image width and height", default=200)
parser.add_argument("-valfrac", "--val-frac", type=float, help="validation fraction", default=0.1)

args = parser.parse_args()






#deltaNum = 1.0e-10  # finite difference stencil width for gradient estimation
#torch.pi = torch.acos(torch.zeros(1)).item() * 2

#
# savePath = '../data/network-parameters/'
# dataPath = '../data/train-data/'
