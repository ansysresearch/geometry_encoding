# sdf-prediction

Predicting the signed distance field for a given geometry


## Network training  
All code in the _src_ directory.

There are three options in network training: (1) training and encoder-decoder network, (2) training  
an enconder-sdf network w/o transfer learning, and (3) training an encoder-sdf network w/ transfer  
learning.

### (1) Training an encoder-decoder network
  The executable script is: _./src/edTrainNetwork.py_  
  The network will learn to encode an input geometry into a compressed state and then to predict the  
    geometry from that encoded state.  
  Options:  
    -a : network architecture id. is searched for in the _./src/networkLib3.py_ file.  
    -d : dataset id. is searched for in the _./data/train-data/_ directory.  
    -n : number of epochs to be trained  
    -s : every how many epochs the state should be saved to a file in the _./data/network-parameters/_  
           directory.  
    -g : which gpu to use  
    -c : case name which is used in the name of the saved network parameter files. can be used to  
           distinguish different setups  
    -f : training fraction. choose a value in (0.0,1.0).  
    -b : batch size  
    -p : print progress dot every how many batches  
  Example:  
    `$ cd ./src`  
    `$ python edTrainNetwork.py -a 38 -d 21 -n 20 -s 5 -g 4 -c test-run-ed -f 0.9 -b 100 -p 5 >> log.txt &`  

### (2) Training an enconder-sdf network w/o transfer learning
  The executable script is: _./src/sdfTrainNetwork.py_  
  The network will learn to encode an input geometry into a compressed state and then to predict the signed  
    distance field from that encoded state and a given location.  
  Options:  
  Example:  
    `$ cd ./src`  
    `$ python sdfTrainNetwork.py -a 68 -d 5 -n 20 -s 5 -g 4 -c test-run-sdf -k 0.1 -w 0.1 -b 125 >> log.txt &`  

### (3) Training an encoder-sdf network w/ transfer learning
  The executable script is: _./src/sdfTrainNetworkTL.py_  
  The network will load an encoder from a file and learn to predict the signed distance field from an  
    encoded state and a given location.  
  Options:  
  Example:  
    `$ cd ./src`  
    `$ python sdfTrainNetworkTL.py -a 68 -d 5 -n 20 -s 5 -g 4 -c test-run-sdf-tl -k 0.1 -w 0.1 >> log.txt &`  
  More details:  
    The loaded encoding network has to match the enconding part of the to-be-trained network. It can be  
      changed in the executable script from line 48 on.  
    
