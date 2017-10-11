from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'keras>=2.0.6',
    'tensorflow-gpu>=1.2',
    'nltk>=3.2.4',
    'numpy>=1.13.1',
    'pandas>=0.20.3',
    'dask>=0.15.3',
    'h5py>=2.7.1',
    'click>=6.7'
]

setup(
    name='hrnn',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=['hrnn', 'hrnn.example'],
    include_package_data=True,
    description='HRNN for text similarity and prediction'
)
