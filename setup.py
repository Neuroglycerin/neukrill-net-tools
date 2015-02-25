#!/usr/bin/env python

from distutils.core import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        import pytest
        pytest.main(self.test_args)

setup(name='neukrill-net',
      version='1.0',
      description='Neukrill-net NDSB tools',
      author='neuroglycerin',
      author_email='root@finlaymagui.re',
      packages=['neukrill_net'],
      tests_require=['pytest'],
      install_requires=['scipy==0.14.0',
                        'numpy==1.9.1',
                        'six==1.8.0',
                        'pytest==2.6.4',
                        'Pillow==2.7.0',
                        'scikit-image==0.10.1',
                        'scikit-learn==0.15.2',
                        'mahotas==1.2.4'],
      cmdclass={'test': PyTest},
)
