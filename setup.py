#!/usr/bin/env python
"""Setup script fo installing the fifipy library."""

from distutils.core import setup

config = {
    'name': 'fifipy',
    'version': '0.4',
    'description': 'FIFI-LS Python library',
    'long_description': 'Collection of programs to reduce FIFI-LS data',
    'author': 'Dario Fadda',
    'author_email': 'darioflute@gmail.com',
    'url': 'https://github.com/darioflute/fifipy.git',
    'download_url': 'https://github.com/darioflute/fifipy',
    'license': 'GPLv3+',
    'packages': ['fifipy','fifipy.cubik'],
    'scripts': ['bin/cubik'],
    'include_package_data': True,
    'package_data': {'fifipy': ['icons/*.png', 'icons/*.gif', 'stylesheet.css',
                                'data/CalibrationResults.csv','data/*.gz','data/*txt','data/*.fits']},
    'install_requires': ['numpy', 'matplotlib', 'astropy','pyqt']
}

setup(**config)
