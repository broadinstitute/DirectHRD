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
        'numpy',
        'pandas',
        'scikit-learn',
        'sigproSS',
        'sigProfilerPlotting',
        'seaborn',
        'matplotlib',
        'SigProfilerMatrixGenerator',
        'SigProfilerExtractor',
        'SigProfilerAssignment'
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
        'Operating System :: OS Independent',
    ],
)

