#!/usr/bin/env python
# coding=utf-8

import codecs
import os
from distutils.core import setup

from setuptools import find_packages

setup_path = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(setup_path, 'README.rst'), encoding='utf-8') as f:
    README = f.read()

setup(name='featselection',
      version='0.2.dev',
      url='https://github.com/rcpf/featselection',
      maintainer='Rogério C. P. Fragoso',
      maintainer_email='rcpf@cin.ufpe.br',
      description='Feature selection methods for Text Classification',
      long_description=README,
      author='Rogério C. P. Fragoso',
      author_email='rcpf@cin.ufpe.br',
      license='MIT',

      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      install_requires=[
          'scikit-learn>=0.19.0',
          'numpy>=1.14.5',
          'pandas>=0.23.3',
          'scipy>=0.13.3',
      ],
      python_requires='>=3',

      packages=find_packages())
