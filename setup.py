"""
-- Created by: Ashok Kumar Pant
-- Email: asokpant@gmail.com
-- Created on: 12/03/2025
"""

from setuptools import setup, find_packages

setup(
    name="aicl434-llms",
    version="1.0.0",
    description="LLMS for AICL434",
    author="Ashok Kumar Pant",
    author_email="asokpant@gmail.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
