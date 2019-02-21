from setuptools import setup, find_packages

setup(
    name = 'MyPackageName',
    version = '1.0.0',
    author = 'Will Harvey',
    author_email = 'wsgh@cs.ubc.ca',
    description = 'Pyro for FOPPL with HMC modified to run batches of chains',
    packages = find_packages(),
)
