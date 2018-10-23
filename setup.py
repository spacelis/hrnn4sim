# from distutils.core import setup
from setuptools import setup
import distutils.cmd
import subprocess

class TarCommand(distutils.cmd.Command):

    """ Tar the project. """

    description = 'Tar the project'
    user_options = [('with-data', None, 'Whether to tar data')]

    def initialize_options(self):
        self.with_data = False

    def finalize_options(self):
        pass

    def run(self):
        command=['/bin/tar', 'czf', 'hrnn.tar.gz', 'hrnn4sim', 'LICENSE', 'README.md', 'setup.py']
        if self.with_data:
            command += ['data/training_set.feather', 'data/valid_set.feather']
            self.announce('Taring the project with data ...')
        else:
            self.announce('Taring the project with NO data ...')
        subprocess.check_call(command)


REQUIRED_PACKAGES = [
    'keras(>=2.0.6)',
    'tensorflow(>=1.2)',
    'nltk(>=3.2.4)',
    'numpy(>=1.13.1)',
    'pandas(>=0.20.3)',
    'dask(>=0.15.3)',
    'h5py(>=2.7.1)',
    'click(>=6.7)',
    'feather-format(>=0.4.0)'
]

setup(
    name='hrnn4sim',
    version='0.1',
    #requires=REQUIRED_PACKAGES,
    install_requires=REQUIRED_PACKAGES,
    packages=['hrnn4sim', 'hrnn4sim.example'],
    description='HRNN for text similarity',
    cmdclass={
        'tar': TarCommand
    }
)
