from setuptools import setup, find_packages

def get_version():
    with open("HRD_classifier/__version__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

setup(
    name='HRD_classifier',
    version=get_version(),
    description='A package for HRD classification',
    author='Ruolin Liu',
    author_email='ruolin@broadinstitute.org',
    packages=setuptools.find_packages(),
    install_requires=[
        # List your dependencies here
        'numpy>=1.21.2,<=1.26.3',
        'pandas==1.5.3',
        'sigProfilerPlotting==1.2.1',
        'matplotlib>=2.2.2',
        'SigProfilerMatrixGenerator==1.2.4',
        'SigProfilerExtractor==1.1.7',
        'SigProfilerAssignment==0.0.5'
        # Add other dependencies
    ],
    include_package_data=True,  # This flag tells setuptools to include package data
    package_data={
        'HRD_classifier': ['data/ID83_model.hrdneg.pickle'],  # Specify the additional files
        'HRD_classifier': ['data/ID83_model.hrdpos.pickle']  # Specify the additional files
    },
    entry_points={
        'console_scripts': [
            'hrd-classifier=HRD_classifier.main:main',  # 'command=module:function'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Academic Software License, 20XX The Broad Institute',
        'Operating System :: OS Independent',
    ],
)

