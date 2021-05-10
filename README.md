# SensingAnomaly

## Dataset.
Downloading Reddit dataset from google drive and put it in the same folder of code.

## Running
To running our model, default parameter is fine. Running python main.py

change --model to 'LOF' or 'GAE' to run different baselines.

change --phase between 'train' and 'test' for trainning and testing the model. Note, LOF only has test phase.

If the train process is interrupted and wanting to continue or wanting to start as test phase, 
--resume_iters need to be assigned the specific value as the last saved iteration number. (Can be found in the checkpoint dir)

--lr is used to control the initial learnning rate, we do not use any rate decay currently. 

--print_net and --use_tensorboard (True or False) are used to print the network architecture and utility of tensorboard.

Full description of parameters can be found in main.py.
