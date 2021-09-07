# coding: utf-8
import setuptools
from setuptools import setup

with open('README.md', 'r') as fp:
    readme = fp.read()
 
VERSION = "1.2.0"
LICENSE = "MIT"

 
setup(
    name='visual_pts',
    version=VERSION,
    description=(
        'Visualize Point Clouds (with bbox) by Plotly'
    ),
    long_description=readme,
    author='JunbinGao, HaoRuan',
    author_email='junbingao@hust.edu.cn',
    maintainer='JunbinGao, HaoRuan',
    maintainer_email='junbingao@hust.edu.cn',
    license=LICENSE,
    packages=setuptools.find_packages(),
    platforms=["all"],
    url='https://github.com/gaojunbin/visual_pts',
    install_requires=[  
        "plotly",  
        "opencv-python",
        "matplotlib",  
        ],  
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
)