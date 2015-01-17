#!/bin/sh
########################################################
#### script will clone repo to directory above this ####
#### and then install requirements, then add the    #### 
#### repo to the Python path of this venv           ####
########################################################
# go up a dir
cd ..
# clone the repo
git clone https://github.com/BVLC/caffe.git
# ON HOLD - see issue

