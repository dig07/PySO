import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EnsembleParticleSwarmOptimization",
    version="0.1.0",
    author="Christopher Moore and Diganta Bandopadhyay",
    author_email="diganta@star.sr.bham.ac.uk",
    description="Ensemble of particle swarms together with various velocity rules for function optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dig07/PySO",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    install_requires=['numpy>=1.15.0', 'scipy', 'pandas', 'matplotlib','scikit-learn','seaborn','pathos','dill','kneed'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)







