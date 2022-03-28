import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PySO",
    version="0.0.1",
    author="Christopher Moore and Diganta Bandopadhyay",
    author_email="cmoore@star.sr.bham.ac.uk",
    description="Python PSO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cjm96/PySO",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)







