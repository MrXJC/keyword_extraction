import setuptools
import keyword_extraction as package


with open("keyword_extraction/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keyword_extraction",
    version=package.__version__,
    author="xujc",
    author_email="",
    description="a tool for keyword_extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=[
        "pandas==0.25.3",
        "jieba==0.42.1",
        "gensim==3.8.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    package_data={'keyword_extraction': []}
)
# -*- coding: utf-8 -*-


