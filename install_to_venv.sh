#!/bin/sh
# ------------------------------------------
# SETUP INSTRUCTIONS
# ------------------------------------------

# cd to your neukrill-net-tools repository
# Either run the script from there, or add cd command to it here
cd path/to/neukrill-net-tools
# e.g.
#cd ~/git/neukrill-net-tools

# Set this parameter manually
# Should be the absolute path to the location you
# want to create as your virtual environment
VIRTUAL_ENV="/absolute/path/to/neukrill-venv"
# e.g.
#VIRTUAL_ENV="/home/username/myvirtualenvs/neukrill-venv"
#VIRTUAL_ENV="/afs/inf.ed.ac.uk/user/sXX/sXXXXXXX/Documents/git/neukrill-venv"
# Caution! Do not use the ~ shorthand!

# ------------------------------------------
# Don't touch the rest unless you know what you're doing
# ------------------------------------------

echo "Installing into virtual environment $VIRTUAL_ENV"

# Make venv
virtualenv --distribute --python=/usr/bin/python2.7 "$VIRTUAL_ENV"

if [ ! -d "$VIRTUAL_ENV" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Virtual environment directory was not created"
  exit 2
fi

# Source it
source "$VIRTUAL_ENV"/bin/activate
pip install numpy
pip install six
pip install -r requirements.txt
python setup.py develop
# Install plotting stuff
pip install ipython[notebook]
pip install matplotlib
pip install https://github.com/ioam/holoviews/archive/v0.8.2.zip
# Install Theano
pip install nose
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Install Pylearn2
cd ../neukrill-net-work/
source start_script
cd ..
git clone https://github.com/lisa-lab/pylearn2.git
cd pylearn2
python setup.py develop
cd ..

# Install OpenCV
wget -qO - https://github.com/Itseez/opencv/archive/2.4.10.1.tar.gz | tar xvz
cd opencv-2.4.10.1
sed -i '50 s/^/#/' cmake/cl2cpp.cmake
mkdir build
cd build
cp /usr/lib64/libpython2.7.so "$VIRTUAL_ENV"/lib
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX="$VIRTUAL_ENV"/local/ -D WITH_TBB=ON -D PYTHON_EXECUTABLE="$VIRTUAL_ENV"/bin/python -D PYTHON_PACKAGES_PATH="$VIRTUAL_ENV"/lib/python2.7/site-packages -D PYTHON_LIBRARY="$VIRTUAL_ENV"/lib/libpython2.7.so -D BUILD_NEW_PYTHON_SUPPORT=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON ..

make
make install

cd ../../
rm -r opencv-2.4.10.1

# Test the installation
cd neukrill-net-tools
python setup.py test

