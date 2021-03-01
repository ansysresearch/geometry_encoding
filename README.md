# Signed Distance Function Prediction


This repo contains the accompnaying code for the paper **Geometry Encoding for Numerical Simulations**. The paper outlines 
  particular requirements of a specific geometry encoding with suitable for carrying out numerical simulations 
  using machine learning. The implementation contains the procedures to generate data and train a combination of 
  neural networks that satisfy these requirements. The geometry inputs are assumed to be black-white square images 
  where the background pixels are 0 and the object pixels are 1. We use Signed Distance Field as a geometry encoder. 
  
  The implementation contains three networks: 
  - *processor*: receives a binary square img as input, and returns the SDF. 
  - *compressor*: receives output of processor and generate an encoding with a 8x reduction in size.
  - *evaluator*: receives the output of processor or compressor, as well as random a list of (x, y) and generates
               the value of SDf at those points.  

## Requirements
see `requirements.txt`

## Data generation  
 The processor and compressor networks are trained using a dataset containing primitive objects including circle, 
 rectangle, diamond and polygon. The analytical expression of the SDF of these primitive geometries are 
 computed on a `[-1,1] x [-1,1]` domain. The initial batch of objects are then augmented in two ways:
  - with a random rotation, scaling or translation
  - combining two or three objects 
 
 When combining multiple SDFs, as discussed [here](https://www.iquilezles.org/www/articles/interiordistance/interiordistance.html), 
 SDFs may not be correct. Therefore, we recomputed the using two euclidean distance transformations. 

 To generate new datasets, run `python main.py -mode data`. 
 
 Some optional parameters when generating data are
 - `--n-obj`: initial number of objects for training. multiplied 9x after augmentation, default=`500`
 - `--img-res`: image resolution, default=`128`
 - `--data-folder`: folder where data are stored, default= `data/datasets/`
 - `--dataset-id`: name of dataset, default= `all128`
 
## training 
For training the processor network run `python main.py -mode train --network-id $NETWORK_ID`. 
The last argument refers to the name of network to be used. See `src/network_lib.py` for the list of available networks.
Recall the processor has to be a UNet.

For training the compressor network run 
`python main.py -mode train --model-flag compressor --network-id $NETWORK_ID_1 --data-network-id $NETWORK_ID_2`.
The argument `--model-flag` is used to indicate this network is a compressor. The argument `--network-id` refers to 
the network name to be used. See `src/network_lib.py` for the list of available networks. Recall the processor 
has to be an AutoEncoder. `--data-network-id` argument refers to an already-trained processor network.  
 
Optional training parameters include:
 - `--network-id`: network id. see `src/network/network_lib.py` for all networks available.
 - `--data-folder`: folder where data are stored, default= `data/datasets/`
 - `--dataset-id`: name of dataset, default= `all`
 - `--data-type`: data type can be `float32` or `float64`. default=`float32`
 - `--n-epochs`: number of epochs, default=`100`
 - `--batch-size`: batch size, default=`10`
 - `--save-every`: save network every x epochs, default=`10`
 - `--loss-fn`: loss function, default=`l1`
 - `--learning-rate`: (initial) learning rate, default=`1e-3`
 - `--min-learning-rate`: minimum value of learning rate, default=`1e-6` 
 - `--plateau-patience`: plateau patience parameter in learning rate scheduling, default=`3`
 - `--plateau-factor`: plateau factor parameter in learning rate scheduling, default=`0.2`
 - `--validation-fraction`: portion of data that is used for validation, default=`0.2`
 - `--save-name`: the tag added to saved models. If not set, the tag will be assigned randomly

Example:

`python main.py -mode train --network-id UNet2 --n-epochs 100 --save-name test1 ` (to train processor)

`python main.py -mode train --model-flag compressor --network-id AE4 --data-network-id UNet2_test1 --save-name test1` (to train compressor)

## test and visualization
We run the model in `test` mode to  generate data on the train, test and exotic datasets.

Optional arguments include:
- `--n-predict`: number of predictions during testing
- `--save-name`: the tag added to saved models

Example:

`python main.py -mode test --network-id UNet2 --save-name test1` (to test processor)

`python main.py -mode test --model-flag compressor --network-id AE4 --save-name test1` (to test compressor)