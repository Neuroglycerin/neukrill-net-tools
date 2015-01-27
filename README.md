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

```
pyvenv my/new/venv/dir
```

On machines __where python 2 is default__ (Ubuntu and DICE machines) you should
run:

```
pyvenv-3.4 my/new/venv/dir
```

__Ubuntu 14.04__ comes with a [broken pyvenv](http://askubuntu.com/questions/488529/pyvenv-3-4-error-returned-non-zero-exit-status-1). We can install a Pyenv without pip, then manually install pip. 

```
# need this as we're using Python 3
sudo apt-get install python3.4-dev
sudo apt-get install python3.4-venv
pyvenv-3.4 --without-pip venv

# source the new venv
source ./venv/bin/activate
wget https://pypi.python.org/packages/source/s/setuptools/setuptools-12.0.4.tar.gz
tar -vzxf setuptools-12.0.4.tar.gz 
cd setuptools-12.0.4/
python setup.py install
cd ..
wget https://pypi.python.org/packages/source/p/pip/pip-6.0.6.tar.gz
tar -vzxf pip-6.0.6.tar.gz
cd pip-6.0.6/
python setup.py install
cd ..
deactivate
```

When you want to work on the project, source it:

```
source my/new/venv/dir/bin/activate
```

Then, _in this repository_ install all the libraries we're 
using to your virtual environment:

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

BLAS Requirements
=================

Most of the neural net packages are going to require BLAS.
You could follow the [Caffe install page][ci] and install
ATLAS.

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
