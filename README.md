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

Then, you should be ready to run the tests.


