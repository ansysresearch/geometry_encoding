# Signed Distance Function Prediction
![](apple.png)

This repo contains implementation of a neural network system for generating accurate signed distance fields (SDF) for 2D geometries. The geometry is assumed to be a black-white square image where the background pixels are 0 and the object pixels are 1. 

## Requirements
see `requirements.txt`

## Parameters
All the parameters of the model are stored, and can be changed, in `params.py`. 

## Data generation  
The learning data are generated from a list of primitive objects including circle, rectangle, diamond and polygon. The analytical expression of the SDF of these primitive geometries are available in `data/geom.py` on a `[-1,1] x [-1,1]` domain. Initially `NUMBER_OBJECT` of these geometries are produced with random sizing and location. This initial set will be augmented in several different ways including  rotation, translation, scaling, rounding (making corners round), adding hole. In addition, more complex geometries that contain 2 or 3 single geometries are added to the dataset. Because of some problems discussed here, SDFs are eventually recomputed using two euclidean distance transformations. 

To generate new datasets, run `generate_data.py`. Remember you can change parameters  in `params.py`.

## Training models
Choose the network structure in `params.py`. For more details about what structures are available, see `network/network_lib.py`. Choose training parameters such as learning rate, batch size, etc in `params.py`. Run `train.py`.

## Testing models
Run `predict.py` for test on train and test datasets. Run `predict_shapes.py` for testing on the exotic shapes. The exotic shapes are more complex shape, that are similar to anything that the network has seen during training. Yet, the models should be able to produce correct SDF values. More "exotic shapes" can be added in `data/exotic_shapes/png_iamges`.

To visulize the model prediction, run `plot_results.py`. This will show `COMPUTE_PREDICTIONS_NUM` examples from the train set, `COMPUTE_PREDICTIONS_NUM` examples from test set, as well as all the exotic shapes predictions. 
