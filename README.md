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

When you want to work on the project, source it:

```
source my/new/venv/dir/bin/activate
```

Then, _in this repository_ install all the libraries we're 
using to your virtual environment:

```
pip install -r requirements.txt
```

This may fail unless you explictly install numpy first: on the travis
deployment the following packages are installed before the rest of the
requirments:

    * pip 
    * numpy 
    * scipy 
    * six

Once all the requirements are installed you can development install (means we don't
have to keep reinstalling the module each time we make a change to the code)
the neukrill-net-tools module by running one of the following when in the tools repo:

```
pip install -e .  
python setup.py develop
```

Once we've solidified our dependencies they will be fully incorporated into 
the setup script and we won't need to worry about the requirements.txt stages
but it is quicker to just do the above for now.


Then, you should be ready to run the tests:
```
python setup.py test
```

