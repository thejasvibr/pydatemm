#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy','scipy','matplotlib','soundfile','tqdm']

test_requirements = [ ]

setup(
    author="Thejasvi Beleyur",
    author_email='thejasvib@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="DATEMM implementation to localise multi-source sounds with reverberation",
    # entry_points={
    #     'console_scripts': [
    #         'cli-ccg=pydatemm:cli_ccg.main',
    #     ],
    # },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pydatemm',
    name='pydatemm',
    packages=find_packages(include=['pydatemm', 'pydatemm.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/thejasvibr/pydatemm',
    version='0.0.1',
    zip_safe=False,
)
