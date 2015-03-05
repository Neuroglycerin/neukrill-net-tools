#!/bin/bash
# 
# Input: the directory to create as the virtual environment
#
# Your neukrill-net-work folder should be in the same directory as neukrill-net-tools
# otherwise this script won't work
# 
#################################################################
#
if [ $# -eq 0 ] || [ -z "$1" ]; then
  echo "Error: No arguments supplied. Need to know target directory."
  exit 2
fi
#################################################################
# Check for necessary commands
#
# Check for virutalenv
if ! hash "virtualenv" 2> /dev/null; then
  echo "This script requires python-virtualenv. Please install it."
  exit 2
fi
# Check for cmake
if ! hash "cmake" 2> /dev/null; then
  echo "This script requires CMake. Please install it."
  exit 2
fi
#
#################################################################
# Need to know the absolute path to the location you
# want to create as your virtual environment
#
# Target venv directory is the argument supplied
# At the moment, this may be a relative location
VIRTUAL_ENV="$1"
#
# Make sure it does not already exist
if [ -d "$VIRTUAL_ENV" ]; then
  # Control will enter here if $VIRTUAL_ENV directory does exist.
  echo "Error: Target directory $VIRTUAL_ENV already exists"
  exit 2
fi
#
echo "Installing into virtual environment $VIRTUAL_ENV"
#
#################################################################
#
# Make venv
virtualenv --no-site-packages -p /usr/bin/python2.7 "$VIRTUAL_ENV"
#
# Make sure we did make venv
if [ ! -d "$VIRTUAL_ENV" ]; then
  # Control will enter here if $VIRTUAL_ENV directory doesn't exist.
  echo "Error: Virtual environment directory was not created"
  exit 2
fi
#
# Change from relative to absolute path
VIRTUAL_ENV="$(cd "$(dirname "$VIRTUAL_ENV")"; pwd)/$(basename "$VIRTUAL_ENV")"
#
echo "Initialised virtual environment at $VIRTUAL_ENV"
#
#################################################################
# Move to the directory of the script we are running now
# This will be the neukrill-net-tools folder, if you haven't moved it!
BASEDIR=$(dirname $0)
cd $BASEDIR
#
#################################################################
# Source virtual environment
source "$VIRTUAL_ENV"/bin/activate
#################################################################
# Install these first
pip install numpy==1.9.1
pip install six==1.8.0
# Then the rest of the requirements
pip install -r requirements.txt
pip install mahotas==1.2.4
#################################################################
# Now development install neukrill-net-tools
python setup.py develop
#################################################################
# Install plotting stuff
pip install ipython[notebook]
pip install matplotlib
pip install git+git://github.com/ioam/param.git@d43a0071823eda175446b591279870e99d2eec67
pip install git+git://github.com/ioam/holoviews.git@v0.8.2
#################################################################
# Install Theano
pip install nose==1.3.4
pip install git+git://github.com/Theano/Theano.git@032a0aa6bc01204e9a3ce8758a1dd97d360562bf
# Install Pylearn2
cd ..
source neukrill-net-work/start_script.sh 0
git clone https://github.com/lisa-lab/pylearn2.git
cd pylearn2
git checkout cf3999e7183f8dcaccccf4dfd2a31bbe3a948a97
git checkout -b neukrillnetchosencommit
python setup.py develop
cd ..
#################################################################
# Install OpenCV
wget -qO - https://github.com/Itseez/opencv/archive/2.4.10.1.tar.gz | tar xvz
cd opencv-2.4.10.1
# Have to command out the line which uses MD5 for DICE version of CMake
sed -i '50 s/^/#/' cmake/cl2cpp.cmake
mkdir build
cd build
#
# Have to copy the python2.7 library file over
if [ -e "$VIRTUAL_ENV"/lib/libpython2.7.so ]; then
    # There is one already there!
    echo "File libpython2.7so is already present"
elif [ -e /usr/lib64/libpython2.7.so ]; then
    # For DICE
    cp /usr/lib64/libpython2.7.so "$VIRTUAL_ENV"/lib
elif [ -e /usr/lib/libpython2.7.so ]; then
    # Other likely library location
    cp /usr/lib/libpython2.7.so "$VIRTUAL_ENV"/lib
elif [ -e /usr/lib/x86_64-linux-gnu/libpython2.7.so ]; then
    # For Ubuntu
    cp /usr/lib/x86_64-linux-gnu/libpython2.7.so "$VIRTUAL_ENV"/lib
fi
if [ ! -e "$VIRTUAL_ENV"/lib/libpython2.7.so ]; then
    echo "Error: Couldn't find libpython2.7.so. Did not install OpenCV to venv."
    exit 2
fi
#
# Now we can run CMake with all these arguments...
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX="$VIRTUAL_ENV"/local/ -D WITH_TBB=ON -D PYTHON_EXECUTABLE="$VIRTUAL_ENV"/bin/python -D PYTHON_PACKAGES_PATH="$VIRTUAL_ENV"/lib/python2.7/site-packages -D PYTHON_LIBRARY="$VIRTUAL_ENV"/lib/libpython2.7.so -D BUILD_NEW_PYTHON_SUPPORT=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON ..
# Install
make
make install
# Remove the downloaded copy of OpenCV - it is in the venv now
cd ../../
rm -r opencv-2.4.10.1
#################################################################
# Test the installation
cd neukrill-net-tools
python setup.py test
#
