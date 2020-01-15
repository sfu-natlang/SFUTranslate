# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='SFUTranslate',
    version='1.0.0',
    description='Neural Machine Translation Toolkit by Natlang Laboratory at SFU',
    long_description=readme,
    install_requires=requirements,
    author='Hassan S. Shavarani',
    author_email='sshavara@sfu.ca',
    url='https://shavarani.github.io/',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'resources'))
)

