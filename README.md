neukrill-net-tools
==================

[![Build Status](https://magnum.travis-ci.com/Neuroglycerin/neukrill-net-tools.svg?token=TAzt1bqxioKxk3ru2s2S)](https://magnum.travis-ci.com/Neuroglycerin/neukrill-net-tools)

Tools coded as part of the NDSB competition.

Virtual Environment
===================

We're using a virtual environment to make sure everyone's using
the same versions (and for reproducibility). To set it up, just run the command
```./install_venv.sh your_new_venv_directory_path```
where `your_new_venv_directory_path` is the directory where you want to create the virtual environment.

The `install_venv.sh` shell script is in the top level of this repository.

**Note 1:** You must have `neukrill-net-tools` and `neukrill-net-work` in the same directory.

**Note 2:** A pylearn2 directory will be created in the same directory as `neukrill-net-tools`.

**Note 3:** You need have `cmake` installed **before** running `install_venv.sh`. You should run
```
sudo apt-get install cmake
```
or
```
yum install cmake
```
or whatever you need for your system to install the `cmake` package.
