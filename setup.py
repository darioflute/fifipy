#!/usr/bin/env python
"""Setup script fo installing the fifipy library."""

from distutils.core import setup
import json

with open('fifipy/version.json') as fp:
    _info = json.load(fp)

config = {
    'name': 'fifipy',
    'version': _info['version'],
    'description': 'FIFI-LS Python library',
    'long_description': 'Collection of programs to reduce FIFI-LS data',
    'author': _info['author'],
    'author_email': 'darioflute@gmail.com',
    'url': 'https://github.com/darioflute/fifipy.git',
    'download_url': 'https://github.com/darioflute/fifipy',
    'python_requires':'>=3.7',
    'license': 'GPLv3+',
    'packages': ['fifipy','fifipy.cubik','fifipy.data'],
    'scripts': ['bin/cubik'],
    'include_package_data': True,
    'package_data': {'fifipy': ['icons/*.png', 'icons/*.gif', 'stylesheet.css',
                                'data/CalibrationResults.csv','data/*.gz','data/*txt',
                                'data/*.fits','version.json']},
}

setup(**config)
