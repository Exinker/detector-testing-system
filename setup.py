from setuptools import find_packages, setup

from detector_testing_system import (
    APPLICATION_DESCRIPTION,
    AUTHOR_EMAIL,
    APPLICATION_NAME,
    APPLICATION_VERSION,
    AUTHOR_NAME,
)


setup(
    # info
    name='_'.join([item.lower() for item in APPLICATION_NAME.split()]),
    description=APPLICATION_DESCRIPTION,
    license='MIT',
    keywords=['spectroscopy', 'detector', 'detector testing'],

    # version
    version=APPLICATION_VERSION,

    # author details
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,

    # setup directories
    packages=find_packages(),

    # setup data
    package_data={
        '': [],
    },

    # requires
    install_requires=[
        item.strip() for item in open('requirements.txt', 'r').readlines()
        if item.strip()
    ],
    python_requires='>=3.10',
)
