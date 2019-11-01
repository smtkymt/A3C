#!/usr/bin/env python

'''
Python distutils setup file for A3C module.

Copyright (C) 2019 Simon D. Levy

MIT License
'''

#from distutils.core import setup
from setuptools import setup

setup (name = 'A3C',
    version = '0.1',
    install_requires = ['tensorflow', 'gym'],
    description = 'Asynchronous Advantage Actor Critic',
    packages = ['a3c'],
    author='Simon D. Levy',
    author_email='simon.d.levy@gmail.com',
    url='https://github.com/simondlevy/A3C',
    license='MIT',
    platforms='Linux; Windows; OS X'
    )
