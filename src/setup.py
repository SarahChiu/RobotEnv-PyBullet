from setuptools import setup, find_packages
import sys

setup(
    name='robotic_env',
    packages=[package for package in find_packages()
        if package.startswith('src')], 
    version='0.1.0',
)
