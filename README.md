# sdf-prediction

Predicting the signed distance field for a given geometry. All code is in the _src_ directory.  

##
## Training data generation  

There are two options in training data generation: (1) generating training data for sdf prediction, 
and (2) generating data for an encoder-decoder network.  

### (1) Generating data for sdf prediction  
  * The executable script is: _./src/geo-sdf.py_  
  * The generator will create two files. The first one contains the binary geometry images, and the  
    second one contains the sampling points for each of the input images.  
  * The geometries are created by sampling a random number of primitive shapes, which are circles,  
    rectangles, triangles, and B-splines.  
  * The script will generate a plot with the binary geometry and the sample point locations as well  
    as a plot with the estimated signed distance field for the last geometry.  
  * There are a couple of parameters that can be set in the executable scipt from line 13 on:  
    - n_geo : the number of geometries to be sampled.  
    - n_samp : the number of sample locations per geometry per epoch.  
    - n_ep_dat : the number of epochs with new data.  
    - n_shape_mu : expected number of primitive shapes per geometry.  
    - n_shape_sigma : standard deviation of number of primitive shapes per geometry.  
    - L_ref : reference size. the size of the sampled primitive shapes is going to scale with L_ref.  
    - minTriAngle : minimum inner angle of sampled triangles.  
    - n_res_x : number of pixels in the x direction.  
    - n_res_y : number of pixels in the y direction.  
    - datasetID : id number used name the files created. this is the number by which datasets can be  
                  loaded during the network training.  
  * Example:  
    `$ cd ./src`  
    `$ python geo-sdf.py >> log.txt`  
  * More parameters: [probably no need to play with | from line 32 on]
    - delta : spacing length scale for point cloud defining geometry surface. during the generation   
              of the dataset, the surface is defined by a point cloud. to estimat the distance for  
              a given location from the surface, the minimum distance of that location to the points  
              from the point cloud is computed.  
    - tol : tolerance value how far the points in the point cloud are allowed to be within the actual  
            geometry. used to avoid issued with truncation errors.  
    - illu_mode : boolean. if true, only one geometry is generated and a higher resolved signed  
                  distance field is plotted. no data is being saved.  
    - print_sampling : boolean. if true, it will print a few details on every sampled primitive shape.  

### (2) Generating daata for an encoder-decoder network  
  * The executable script is: _./src/geo-ed.py_  
  * The generator will create one file containing the binary geometry images.  
  * The geometries are created by sampling a random number of primitive shapes, which are circles,  
    rectangles, triangles, and B-splines.  
  * The script will generate a plot with the binary geometry for the last geometry.  
  * There are a couple of parameters that can be set in the executable scipt from line 13 on:  
    - n_geo : the number of geometries to be sampled.  
    - n_shape_mu : expected number of primitive shapes per geometry.  
    - n_shape_sigma : standard deviation of number of primitive shapes per geometry.  
    - L_ref : reference size. the size of the sampled primitive shapes is going to scale with L_ref.  
    - minTriAngle : minimum inner angle of sampled triangles.  
    - n_res_x : number of pixels in the x direction.  
    - n_res_y : number of pixels in the y direction.  
    - datasetID : id number used name the files created. this is the number by which datasets can be  
                  loaded during the network training.  
  * Example:  
    `$ cd ./src`  
    `$ python geo-ed.py >> log.txt`  
  * More parameters: [probably no need to play with | from line 30 on]
    - delta : spacing length scale for point cloud defining geometry surface. during the generation   
              of the dataset, the surface is defined by a point cloud. to estimat the distance for  
              a given location from the surface, the minimum distance of that location to the points  
              from the point cloud is computed.  
    - tol : tolerance value how far the points in the point cloud are allowed to be within the actual  
            geometry. used to avoid issued with truncation errors.  
    - print_sampling : boolean. if true, it will print a few details on every sampled primitive shape.  


## Network training  

There are three options in network training: (1) training and encoder-decoder network, (2) training  
an enconder-sdf network w/o transfer learning, and (3) training an encoder-sdf network w/ transfer  
learning.

### (1) Training an encoder-decoder network
  * The executable script is: _./src/edTrainNetwork.py_  
  * The network will learn to encode an input geometry into a compressed state and then to predict the  
    geometry from that encoded state.  
  * Options:  
    - -a : network architecture id. is searched for in the _./src/networkLib3.py_ file.  
    - -d : dataset id. is searched for in the _./data/train-data/_ directory.  
    - -n : number of epochs to be trained.  
    - -s : every how many epochs the state should be saved to a file in the _./data/network-parameters/_  
           directory.  
    - -g : which gpu to use.  
    - -c : case name which is used in the name of the saved network parameter files. can be used to  
           distinguish different setups.  
    - -f : training fraction. choose a value in (0.0,1.0).  
    - -b : batch size.  
    - -p : print progress dot every how many batches.  
    - -l : learning rate.  
  * Example:  
    `$ cd ./src`  
    `$ python edTrainNetwork.py -a 38 -d 21 -n 20 -s 5 -g 4 -c test-run-ed -f 0.9 -b 100 -p 5 >> log.txt &`  

### (2) Training an enconder-sdf network w/o transfer learning
  * The executable script is: _./src/sdfTrainNetwork.py_  
  * The network will learn to encode an input geometry into a compressed state and then to predict the signed  
    distance field from that encoded state and a given location.  
  * Options:  
    - -a : network architecture id. is searched for in the _./src/networkLib3.py_ file.  
    - -d : dataset id. is searched for in the _./data/train-data/_ directory.  
    - -n : number of epochs to be trained.  
    - -s : every how many epochs the state should be saved to a file in the _./data/network-parameters/_  
           directory.  
    - -g : which gpu to use.  
    - -c : case name which is used in the name of the saved network parameter files. can be used to  
           distinguish different setups.  
    - -f : training fraction. choose a value in (0.0,1.0).  
    - -b : batch size.  
    - -p : print progress dot every how many batches.  
    - -l : learning rate.  
    - -k : eikonal loss term factor. if unspecified or zero, not eikonal loss will be used.  
    - -w : weighted L1 loss shape width. if unspecified or zero, not weighted L1 loss will be used.  
  * Example:  
    `$ cd ./src`  
    `$ python sdfTrainNetwork.py -a 68 -d 5 -n 20 -s 5 -g 4 -c test-run-sdf -k 0.1 -w 0.1 -b 125 >> log.txt &`  
  * More details:  
    - The weighting function for the weighted L1 loss can be modified in the file _./src/SdfTrain.py_. It must be  
      modified twice, for the training loss and for the test loss computations. Those modifications are made from  
      line 65 on and from line 97 on, respectively. In the code there two different shapes implemented already,  
      one with an exponential decay, and one with a cosine and exponential decay term.  
    - The stencil half-width of the finite difference derivative calculations of the eikonal loss term is defined 
      in _./src/parseArgs.py_ in line 79.

### (3) Training an encoder-sdf network w/ transfer learning
  * The executable script is: _./src/sdfTrainNetworkTL.py_  
  * The network will load an encoder from a file and learn to predict the signed distance field from an  
    encoded state and a given location.  
  * Options:  
    - -a : network architecture id. is searched for in the _./src/networkLib3.py_ file.  
    - -d : dataset id. is searched for in the _./data/train-data/_ directory.  
    - -n : number of epochs to be trained.  
    - -s : every how many epochs the state should be saved to a file in the _./data/network-parameters/_  
           directory.  
    - -g : which gpu to use.  
    - -c : case name which is used in the name of the saved network parameter files. can be used to  
           distinguish different setups.  
    - -f : training fraction. choose a value in (0.0,1.0).  
    - -b : batch size.  
    - -p : print progress dot every how many batches.  
    - -l : learning rate.  
    - -k : eikonal loss term factor. if unspecified or zero, not eikonal loss will be used.  
    - -w : weighted L1 loss shape width. if unspecified or zero, not weighted L1 loss will be used.  
  * Example:  
    `$ cd ./src`  
    `$ python sdfTrainNetworkTL.py -a 68 -d 5 -n 20 -s 5 -g 4 -c test-run-sdf-tl -k 0.1 -w 0.1 >> log.txt &`  
  * More details:  
    - The loaded encoding network has to match the enconding part of the to-be-trained network. It can be  
      changed in the executable script from line 48 on.  
    - The weighting function for the weighted L1 loss can be modified in the file _./src/SdfTrain.py_. It must be  
      modified twice, for the training loss and for the test loss computations. Those modifications are made from  
      line 65 on and from line 97 on, respectively. In the code there two different shapes implemented already,  
      one with an exponential decay, and one with a cosine and exponential decay term.  
    - The stencil half-width of the finite difference derivative calculations of the eikonal loss term is defined 
      in _./src/parseArgs.py_ in line 79.
    
