from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'keras>=2.0.6',
    'tensorflow>=1.2.1',
    'nltk>=3.2.4',
    'numpy>=1.13.1',
    'pandas>=0.20.3',
    'click>=6.7'
]

setup(
    name='hrnn4sim',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    description='HRNN for text similarity'
)
