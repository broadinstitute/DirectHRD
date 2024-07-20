from setuptools import setup, find_packages

setup(
    name='HRD_classifier',
    version='0.1.0',
    description='A package for HRD classification',
    author='Ruolin Liu',
    author_email='ruolin@broadinstitute.org',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'numpy==1.22.0',
        'pandas==1.5.3',
        'scikit-learn',
        'sigproSS',
        'sigProfilerPlotting==1.2.1',
        'seaborn',
        'matplotlib',
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

