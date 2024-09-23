__author__ = 'Brian M Anderson'
# Created on 9/15/2020


from setuptools import setup
import os

# Get the directory where the setup.py script is located
this_directory = os.path.abspath(os.path.dirname(__file__))

# Construct the full path to the requirements.txt file
requirements_path = os.path.join(this_directory, 'requirements.txt')

with open("README.md", "r") as fh:
    long_description = fh.read()
with open(requirements_path) as f:
    required = f.read().splitlines()

setup(
    name='PlotScrollNumpyArrays',
    author='Brian Mark Anderson',
    author_email='markba122@gmail.com',
    version='2.0.11',
    description='Services for plotting and viewing',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'PlotScrollNumpyArrays': 'src/PlotScrollNumpyArrays'},
    packages=['PlotScrollNumpyArrays'],
    include_package_data=True,
    url='https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
    ],
    install_requires=required,
)