from setuptools import setup, find_packages

setup(
    name='ArbitryonMasking',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'autodistill-grounded-sam-2',
        'opencv-python-headless',
        'matplotlib',
        'numpy',
    ],
)