__author__ = 'Brian M Anderson'
# Created on 9/15/2020


from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='DicomRTTool',
    author='Brian Mark Anderson',
    author_email='markba122@gmail.com',
    version='2.0.0',
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