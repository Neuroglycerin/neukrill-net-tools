neukrill-net-tools
==================

[![Build Status](https://magnum.travis-ci.com/Neuroglycerin/neukrill-net-tools.svg?token=TAzt1bqxioKxk3ru2s2S)](https://magnum.travis-ci.com/Neuroglycerin/neukrill-net-tools)

Tools coded as part of the NDSB competition.

Virtual Environment
===================

We're using a virtual environment to make sure everyone's using
the same versions (and for reproducibility). To set it up, 
use the pyvenv script (should come with Python 3). Go _somewhere
else_ and make a venv directory:

Python 3
--------

We're __moving away from Python 3__, as it seems to be easier to work with 
Pylearn2 or Theano in Python 2.7. If you're setting up for the first time,
skip to the next section on Python 2.

```
pyvenv path/to/neukrill-venv
```

On machines __where python 2 is default__ (Ubuntu and DICE machines) you should
run:

```
pyvenv-3.4 path/to/neukrill-venv
```

__Ubuntu 14.04__ comes with a [broken pyvenv](http://askubuntu.com/questions/488529/pyvenv-3-4-error-returned-non-zero-exit-status-1). We can install a Pyenv without pip, then manually install pip. 

```
# need this as we're using Python 3
sudo apt-get install python3.4-dev
sudo apt-get install python3.4-venv
pyvenv-3.4 --without-pip path/to/neukrill-venv

# source the new venv
source path/to/neukrill-venv/bin/activate
wget -qO - https://pypi.python.org/packages/source/s/setuptools/setuptools-12.0.4.tar.gz | tar xvz
cd setuptools-12.0.4/
python setup.py install
cd ..
wget -qO - https://pypi.python.org/packages/source/p/pip/pip-6.0.6.tar.gz | tar xvz
cd pip-6.0.6/
python setup.py install
cd ..
deactivate
rm -r setuptools-12.0.4
rm -r cd pip-6.0.6
```

Python 2.7 virtualenv
---------------------

To make a __Python 2.7__ virtual environment instead of 3.4, we can follow 
Edinburgh University's [own instructions on this][is] 
(_do this on any DICE machine_):

```
virtualenv --distribute --python=/usr/bin/python2.7 path/to/neukrill-venv-py2.7
```

OR:

```
virtualenv -p /usr/bin/python2.7 path/to/neukrill-venv-py2.7
```

Activating
----------

When you want to work on the project, source it:

```
source path/to/neukrill-venv/bin/activate
```

Then, _in this repository_ install all the libraries we're 
using to your virtual environment:
```
cd path/to/neukrill-net-tools
```

```
# to prevent failure, install these two first
pip install numpy
pip install six

pip install -r requirements.txt
```

On the travis deployment the following packages are installed before the rest of the
requirements:

* pip 
* numpy 
* scipy 
* six

Once all the requirements are installed you can development install (means we don't
have to keep reinstalling the module each time we make a change to the code)
the neukrill-net-tools module by running one of the following when in the tools repo:

```
pip install -e .
```

__or__

```
python setup.py develop
```

Once we've solidified our dependencies they will be fully incorporated into 
the setup script and we won't need to worry about the requirements.txt stages
but it is quicker to just do the above for now.


Then, you should be ready to run the tests:
```
python setup.py test
```

If you want to use IPython (and its notebook) in this virtualenv you'll have to 
install it as well:

```
pip install ipython[notebook]
```

But be careful, if you run IPython in that virtualenv before installing as 
above, you can end up working in [mixed environments][ipy], which would be
unpleasant.

And you might also want to plot things, and that's usually done using
matplotlib so:

```
pip install matplotlib
```

That'll also install a bunch of requirements.

[ipy]: https://coderwall.com/p/xdox9a/running-ipython-cleanly-inside-a-virtualenv

BLAS Requirements
=================

Most of the neural net packages are going to require BLAS. However, most of the
machines we're working with this stuff on should already have this set up.

For Arch Linux
--------------

Was not seeing any speedup on multiple cores with default [BLAS for Arch Linux][ab].
Installed OpenBLAS through the linuxfr repository (installed to get yaourt) and
immediately saw massive speedup. Would recommend installing this on Arch.

Ubuntu
------

You may want to install OpenBLAS or ATLAS through apt-get.

[ci]: http://caffe.berkeleyvision.org/installation.html
[ab]: http://www.netlib.org/lapack/

Installing Pylearn2
===================

First, make sure Theano and nose are installed using pip:

```
pip install nose
```

For Theano, you might have problems without the [bleeding edge version][tb]:

```
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
```

To be safe (I don't know if this would break it or not), run the start script
in the work repository to set up the environment variables.

```
cd ../neukrill-net-work/
source start_script
```

We now clone the their [github repo][pylearn2], and do a development install, as we 
did with our own tools repository (still inside the virtual environment).

```
cd ..
git clone https://github.com/lisa-lab/pylearn2.git
cd pylearn2
python setup.py develop
cd ..
```

It must be kept up to date manually by pulling the repository, but it probably
won't go out of date within this project.

[pylearn2]: https://github.com/lisa-lab/pylearn2
[tb]: http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions

Installing OpenCV
===================


You will need >1GB free space to install OpenCV. When the process is finished, it will take up about 300MB.

First, you need to install `cmake` if you don't have it already. Do this without being in the virtual environment.

On Debian:
```
sudo apt-get install cmake
```
On CentOS:
```
yum install cmake
```

Setup:
```
wget -qO - https://github.com/Itseez/opencv/archive/3.0.0-beta.tar.gz | tar xvz
cd opencv-3.0.0-beta
mkdir build
cd build
```

Now set a variable for the path your venv is at, and source it. Make sure it is an absolute path, not a relative path.
This is just for the purposes of making the install process easier.
```
$VIRTUAL_ENV = '/absolute/path/to/neukrill-venv'
source $VIRTUAL_ENV/bin/activate
```

For a Python 3.4 venv, do the following:
```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/ -D PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python3.4 -D PYTHON_PACKAGES_PATH=$VIRTUAL_ENV/lib/python3.4/site-packages -D BUILD_NEW_PYTHON_SUPPORT=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON ..
```

Alternatively, to install on a Python 2.7 venv:
```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/local/ -D PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python -D PYTHON_PACKAGES_PATH=$VIRTUAL_ENV/lib/python2.7/site-packages -D BUILD_NEW_PYTHON_SUPPORT=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON ..
```

After doing `cmake` for either Python 2.7 or 3.4, scroll up and check that the directories are correct and point at the venv in the Python 2/3 section, and that `Python (for build)` correctly points to the venv as well.

Finish:
```
make -j
make install
```
Note: The `make` step may take about half an hour when using only a single core. The above code uses all available cores.

Now let's try it out and check it is working:
```
python
import cv2
```
This should import without error.

If this is successful, you can now remove the copy of OpenCV you downloaded.
```
cd ../../
rm -r opencv-3.0.0-beta
```
