from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fluxions",
    version="2.0.0",
    author="Harvard CS207 Final Project Group 10",
    description="A package for Automatic Differentiation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CS207-Final-Project-Group-10/cs207-FinalProject",
    tests_require=["pytest"],
    packages=['fluxions'],
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)




